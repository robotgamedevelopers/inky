// Soak clients for `inky serve`. Shell launches this through nova-secrets exec so
// INKY_SERVE_KEY is in the environment only. This process never prints that value
// and never puts it on argv.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	neturl "net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	lenShort  = 64
	lenMedium = 1024
	lenLong   = 4096
)

var promptKinds = []struct {
	name string
	n    int
}{
	{"short", lenShort},
	{"medium", lenMedium},
	{"long", lenLong},
}

var maxTokensChoices = []int{16, 64, 256}

func prompt(kind, n int) string {
	unit := "The soak client asks for a brief note about the ocean. "
	var b strings.Builder
	b.WriteString("prompt-length ")
	b.WriteString(promptKinds[kind].name)
	b.WriteByte('\n')
	for b.Len() < n {
		b.WriteString(unit)
	}
	s := b.String()
	if len(s) > n {
		s = s[:n]
	}
	return s
}

// withID keeps the byte length and makes some prompts miss the lane prefix cache.
func withID(s string, id uint64) string {
	tag := fmt.Sprintf(" id=%08d", id%100000000)
	if len(tag) >= len(s) {
		return tag[:len(s)]
	}
	return s[:len(s)-len(tag)] + tag
}

type dialBox struct{ c atomic.Value }

type ctxKey struct{}

func (b *dialBox) conn() net.Conn {
	v := b.c.Load()
	if v == nil {
		return nil
	}
	return v.(net.Conn)
}

type chatReq struct {
	Model       string    `json:"model"`
	Messages    []chatMsg `json:"messages"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature float64   `json:"temperature"`
	Stream      bool      `json:"stream"`
	Seed        uint64    `json:"seed"`
}

type chatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type totals struct {
	requests     atomic.Int64
	http200      atomic.Int64
	http4xx      atomic.Int64
	http401      atomic.Int64
	http429      atomic.Int64
	http5xx      atomic.Int64
	unexpected   atomic.Int64
	completed    atomic.Int64
	streamErrors atomic.Int64
	disconnected atomic.Int64
	transport    atomic.Int64
	timeouts     atomic.Int64
	inflight     atomic.Int64
	inflightMax  atomic.Int64
}

func (t *totals) enter() {
	n := t.inflight.Add(1)
	for {
		old := t.inflightMax.Load()
		if n <= old || t.inflightMax.CompareAndSwap(old, n) {
			return
		}
	}
}

func (t *totals) note(code int) {
	switch {
	case code == 200:
		t.http200.Add(1)
	case code == 401:
		t.http401.Add(1)
		t.http4xx.Add(1)
	case code == 429:
		t.http429.Add(1)
		t.http4xx.Add(1)
	case code >= 500:
		t.http5xx.Add(1)
	case code >= 400:
		t.http4xx.Add(1)
		t.unexpected.Add(1)
	default:
		t.unexpected.Add(1)
	}
}

func main() {
	url := flag.String("url", "http://127.0.0.1:18090/v1/chat/completions", "chat completions URL")
	duration := flag.Int("duration", 3600, "seconds to open new requests")
	clients := flag.Int("clients", 24, "concurrent clients")
	lanes := flag.Int("lanes", 16, "server lane count, recorded in the summary")
	queue := flag.Int("queue", 64, "server queue depth, recorded in the summary")
	seed := flag.Uint64("seed", 1, "RNG seed")
	disconnect := flag.Int("disconnect-pct", 30, "percent of requests closed mid-response")
	selfCheck := flag.Bool("self-check", false, "check prompt lengths and exit")
	flag.Parse()

	if *selfCheck {
		for i, k := range promptKinds {
			p := prompt(i, k.n)
			if len(p) != k.n {
				fmt.Fprintf(os.Stderr, "prompt %s length %d want %d\n", k.name, len(p), k.n)
				os.Exit(1)
			}
			q := withID(p, 42)
			if len(q) != k.n {
				fmt.Fprintf(os.Stderr, "prompt %s id length %d want %d\n", k.name, len(q), k.n)
				os.Exit(1)
			}
		}
		fmt.Printf("prompt_short_bytes=%d\n", lenShort)
		fmt.Printf("prompt_medium_bytes=%d\n", lenMedium)
		fmt.Printf("prompt_long_bytes=%d\n", lenLong)
		fmt.Printf("max_tokens=%d,%d,%d\n", maxTokensChoices[0], maxTokensChoices[1], maxTokensChoices[2])
		fmt.Println("self_check=ok")
		return
	}
	if *duration < 1 || *clients < 1 || *lanes < 1 || *queue < 0 || *disconnect < 0 || *disconnect > 100 {
		fmt.Fprintln(os.Stderr, "duration, clients, or disconnect-pct is out of range")
		os.Exit(2)
	}
	key := os.Getenv("INKY_SERVE_KEY")
	if key == "" {
		fmt.Fprintln(os.Stderr, "INKY_SERVE_KEY is not set")
		os.Exit(2)
	}

	dialer := &net.Dialer{Timeout: 10 * time.Second}
	transport := &http.Transport{
		Proxy: nil,
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			c, err := dialer.DialContext(ctx, network, addr)
			if err != nil {
				return nil, err
			}
			if b, ok := ctx.Value(ctxKey{}).(*dialBox); ok {
				b.c.Store(c)
			}
			return c, nil
		},
		DisableKeepAlives:   true,
		MaxIdleConns:        0,
		IdleConnTimeout:     1 * time.Second,
		TLSHandshakeTimeout: 5 * time.Second,
		// localhost only; do not follow a redirect that might carry the bearer
	}
	httpClient := &http.Client{
		Transport: transport,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}

	var tot totals
	deadline := time.Now().Add(time.Duration(*duration) * time.Second)
	var wg sync.WaitGroup
	ctxRun, cancelRun := context.WithCancel(context.Background())
	defer cancelRun()

	go func() {
		tick := time.NewTicker(30 * time.Second)
		defer tick.Stop()
		for {
			select {
			case <-ctxRun.Done():
				return
			case <-tick.C:
				fmt.Fprintf(os.Stderr, "soak tick requests=%d inflight=%d 5xx=%d disconnected=%d\n",
					tot.requests.Load(), tot.inflight.Load(), tot.http5xx.Load(), tot.disconnected.Load())
			}
		}
	}()

	for w := 0; w < *clients; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(*seed, uint64(w)+1))
			for time.Now().Before(deadline) && ctxRun.Err() == nil {
				one(&tot, httpClient, key, *url, *disconnect, *seed, rng)
			}
		}(w)
	}
	wg.Wait()
	cancelRun()

	inflightEnd := tot.inflight.Load()
	probeOK := 0
	if inflightEnd == 0 {
		var probes sync.WaitGroup
		var ok atomic.Int64
		start := make(chan struct{})
		for i := 0; i < *lanes; i++ {
			probes.Add(1)
			go func(i int) {
				defer probes.Done()
				<-start
				if probe(httpClient, key, *url, *seed+uint64(i)) {
					ok.Add(1)
				}
			}(i)
		}
		close(start)
		probes.Wait()
		probeOK = int(ok.Load())
	}
	serverActiveLanes, serverInflight, serverHealthOK := checkHealthDrain(httpClient, healthURL(*url))

	fmt.Printf("seed=%d\n", *seed)
	fmt.Printf("duration_s=%d\n", *duration)
	fmt.Printf("clients=%d\n", *clients)
	fmt.Printf("lanes=%d\n", *lanes)
	fmt.Printf("queue=%d\n", *queue)
	fmt.Printf("disconnect_pct=%d\n", *disconnect)
	fmt.Printf("prompt_short_bytes=%d\n", lenShort)
	fmt.Printf("prompt_medium_bytes=%d\n", lenMedium)
	fmt.Printf("prompt_long_bytes=%d\n", lenLong)
	fmt.Printf("max_tokens=%d,%d,%d\n", maxTokensChoices[0], maxTokensChoices[1], maxTokensChoices[2])
	fmt.Printf("requests=%d\n", tot.requests.Load())
	fmt.Printf("http_200=%d\n", tot.http200.Load())
	fmt.Printf("http_4xx=%d\n", tot.http4xx.Load())
	fmt.Printf("http_401=%d\n", tot.http401.Load())
	fmt.Printf("http_429=%d\n", tot.http429.Load())
	fmt.Printf("http_5xx=%d\n", tot.http5xx.Load())
	fmt.Printf("disconnected=%d\n", tot.disconnected.Load())
	fmt.Printf("transport_errors=%d\n", tot.transport.Load())
	fmt.Printf("timeouts=%d\n", tot.timeouts.Load())
	fmt.Printf("inflight_end=%d\n", inflightEnd)
	fmt.Printf("inflight_max=%d\n", tot.inflightMax.Load())
	fmt.Printf("server_active_lanes_end=%d\n", serverActiveLanes)
	fmt.Printf("server_inflight_end=%d\n", serverInflight)
	fmt.Printf("server_health_ok=%d\n", boolInt(serverHealthOK))
	fmt.Printf("workload_completed=%d\n", tot.completed.Load())
	fmt.Printf("unexpected_status=%d\n", tot.unexpected.Load())
	fmt.Printf("stream_errors=%d\n", tot.streamErrors.Load())
	fmt.Printf("recovery_completed=%d\n", probeOK)
	fmt.Printf("recovery_requested=%d\n", *lanes)
	fmt.Println("inflight_source=server_health_and_client_outstanding")

	if inflightEnd != 0 || !serverHealthOK || serverActiveLanes != 0 || serverInflight != 0 || tot.http5xx.Load() != 0 || tot.http401.Load() != 0 || tot.unexpected.Load() != 0 || tot.streamErrors.Load() != 0 || tot.completed.Load() == 0 || tot.transport.Load() != 0 || tot.timeouts.Load() != 0 || probeOK != *lanes {
		os.Exit(1)
	}
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func healthURL(chatURL string) string {
	u, err := neturl.Parse(chatURL)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return strings.TrimRight(chatURL, "/") + "/health"
	}
	u.Path = "/health"
	u.RawQuery = ""
	u.Fragment = ""
	return u.String()
}

type healthSnapshot struct {
	OK          bool `json:"ok"`
	ActiveLanes *int `json:"active_lanes"`
	Inflight    *int `json:"inflight"`
}

func checkHealthDrain(httpClient *http.Client, url string) (activeLanes int, inflight int, ok bool) {
	deadline := time.Now().Add(2 * time.Second)
	activeLanes, inflight = -1, -1
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err == nil {
			req.Close = true
			resp, doErr := httpClient.Do(req)
			if doErr == nil {
				var snap healthSnapshot
				decErr := json.NewDecoder(resp.Body).Decode(&snap)
				_ = resp.Body.Close()
				if resp.StatusCode == http.StatusOK && decErr == nil && snap.OK && snap.ActiveLanes != nil && snap.Inflight != nil {
					activeLanes = *snap.ActiveLanes
					inflight = *snap.Inflight
					ok = true
					cancel()
					if activeLanes == 0 && inflight == 0 {
						return activeLanes, inflight, true
					}
				} else {
					cancel()
				}
			} else {
				cancel()
			}
		} else {
			cancel()
		}
		if time.Now().After(deadline) {
			return activeLanes, inflight, ok
		}
		time.Sleep(25 * time.Millisecond)
	}
}

func one(tot *totals, httpClient *http.Client, key, url string, disconnectPct int, seed uint64, rng *rand.Rand) {
	tot.enter()
	defer tot.inflight.Add(-1)
	id := uint64(tot.requests.Add(1))

	kind := int(rng.UintN(uint(len(promptKinds))))
	bodyText := prompt(kind, promptKinds[kind].n)
	if rng.IntN(2) == 0 {
		bodyText = withID(bodyText, id)
	}
	maxTok := maxTokensChoices[rng.IntN(len(maxTokensChoices))]
	disconnect := rng.IntN(100) < disconnectPct
	reqSeed := seed ^ (id * 0x9E3779B97F4A7C15)

	payload, err := json.Marshal(chatReq{
		Model:       "inkling-2-mini",
		Messages:    []chatMsg{{Role: "user", Content: bodyText}},
		MaxTokens:   maxTok,
		Temperature: 0,
		Stream:      true,
		Seed:        reqSeed,
	})
	if err != nil {
		tot.transport.Add(1)
		return
	}

	box := &dialBox{}
	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
	defer cancel()
	ctx = context.WithValue(ctx, ctxKey{}, box)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		tot.transport.Add(1)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	req.Close = true

	resp, err := httpClient.Do(req)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			tot.timeouts.Add(1)
		} else {
			tot.transport.Add(1)
		}
		return
	}
	defer resp.Body.Close()
	tot.note(resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		return
	}

	if disconnect {
		if c := box.conn(); c != nil {
			_ = c.SetReadDeadline(time.Now().Add(5 * time.Second))
		}
		buf := make([]byte, 512)
		_, _ = resp.Body.Read(buf)
		hold := time.Duration(50+rng.IntN(1450)) * time.Millisecond
		timer := time.NewTimer(hold)
		<-timer.C
		if c := box.conn(); c != nil {
			_ = c.Close()
		}
		tot.disconnected.Add(1)
		return
	}

	err = validateStream(resp.Body)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			tot.timeouts.Add(1)
		} else {
			tot.streamErrors.Add(1)
		}
		return
	}
	tot.completed.Add(1)
}

// validateStream requires both an application finish reason and the SSE terminator.
// A successful HTTP status alone does not mean generation completed.
func validateStream(r io.Reader) error {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 4096), 1024*1024)
	finished, done := false, false
	var dataLines []string
	for s.Scan() {
		line := s.Text()
		if line != "" {
			if strings.HasPrefix(line, "data:") {
				dataLines = append(dataLines, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
			}
			continue
		}
		if len(dataLines) == 0 {
			continue
		}
		data := strings.Join(dataLines, "\n")
		dataLines = nil
		if done {
			return fmt.Errorf("data after DONE")
		}
		if data == "[DONE]" {
			done = true
			continue
		}
		var event struct {
			Error   json.RawMessage `json:"error"`
			Choices []struct {
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return err
		}
		if len(event.Error) > 0 && string(event.Error) != "null" {
			return fmt.Errorf("stream error")
		}
		for _, choice := range event.Choices {
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				finished = true
			}
		}
	}
	if len(dataLines) > 0 {
		return fmt.Errorf("unterminated SSE event")
	}
	if err := s.Err(); err != nil {
		return err
	}
	if !finished || !done {
		return fmt.Errorf("incomplete stream")
	}
	return nil
}

func probe(httpClient *http.Client, key, url string, seed uint64) bool {
	payload, err := json.Marshal(chatReq{
		Model: "inkling-2-mini", Messages: []chatMsg{{Role: "user", Content: prompt(0, lenShort)}},
		MaxTokens: 8, Temperature: 0, Stream: true, Seed: seed,
	})
	if err != nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	req.Close = true
	resp, err := httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK && validateStream(resp.Body) == nil
}
