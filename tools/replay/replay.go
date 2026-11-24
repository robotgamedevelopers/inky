package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ResponseMetrics mirrors src/record.rs response metrics.
type ResponseMetrics struct {
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	TTFTMs           uint64 `json:"ttft_ms"`
	TotalMs          uint64 `json:"total_ms"`
	FinishReason     string `json:"finish_reason"`
	StatusCode       int    `json:"status_code"`
}

// RecordedEntry mirrors one JSONL line written by `inky serve --record-dir`.
type RecordedEntry struct {
	SchemaVersion uint32          `json:"schema_version"`
	RecordedAtMs  uint64          `json:"recorded_at_ms"`
	CardID        string          `json:"card_id"`
	TurnIndex     uint64          `json:"turn_index"`
	Endpoint      string          `json:"endpoint"`
	Request       json.RawMessage `json:"request"`
	Response      ResponseMetrics `json:"response"`
}

// Card groups all sequential turns belonging to one X-Inky-Card-Id conversation.
type Card struct {
	ID    string
	Turns []RecordedEntry
}

// Clock abstracts time and sleep for deterministic socket/httptest unit tests.
type Clock interface {
	Now() time.Time
	Sleep(d time.Duration)
}

type realClock struct{}

func (realClock) Now() time.Time        { return time.Now() }
func (realClock) Sleep(d time.Duration) { time.Sleep(d) }

// SweepConfig configures a replay lane-count sweep.
type SweepConfig struct {
	BaseURL        string
	APIKey         string
	LaneCounts     []int
	CardsPerRung   int           // 0 = max(len(cards), lanes)
	Duration       time.Duration // 0 = fixed card count mode
	MaxRetries     int
	HealthInterval time.Duration
	Knee           KneeConfig
	Client         *http.Client
	Clock          Clock
}

// LoadReplaySet loads one .jsonl file or a directory of .jsonl files, groups turns
// by card_id in turn_index order, and returns cards in deterministic order.
func LoadReplaySet(path string) ([]Card, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	var files []string
	if info.IsDir() {
		entries, err := os.ReadDir(path)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl") {
				files = append(files, filepath.Join(path, e.Name()))
			}
		}
		sort.Strings(files)
	} else {
		files = []string{path}
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no .jsonl files found in %s", path)
	}

	var all []RecordedEntry
	for _, f := range files {
		entries, err := loadJSONLFile(f)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		all = append(all, entries...)
	}
	if len(all) == 0 {
		return nil, fmt.Errorf("no valid replay entries found in %s", path)
	}

	grouped := make(map[string][]RecordedEntry)
	for _, e := range all {
		grouped[e.CardID] = append(grouped[e.CardID], e)
	}

	cards := make([]Card, 0, len(grouped))
	for id, turns := range grouped {
		sort.SliceStable(turns, func(i, j int) bool {
			if turns[i].TurnIndex != turns[j].TurnIndex {
				return turns[i].TurnIndex < turns[j].TurnIndex
			}
			return turns[i].RecordedAtMs < turns[j].RecordedAtMs
		})
		cards = append(cards, Card{ID: id, Turns: turns})
	}
	sort.Slice(cards, func(i, j int) bool {
		if cards[i].Turns[0].RecordedAtMs != cards[j].Turns[0].RecordedAtMs {
			return cards[i].Turns[0].RecordedAtMs < cards[j].Turns[0].RecordedAtMs
		}
		return cards[i].ID < cards[j].ID
	})
	return cards, nil
}

func loadJSONLFile(path string) ([]RecordedEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []RecordedEntry
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 16*1024*1024)
	lineNum := 0
	for sc.Scan() {
		lineNum++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var e RecordedEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNum, err)
		}
		if len(bytes.TrimSpace(e.Request)) == 0 {
			continue
		}
		// Skip malformed 400 client requests captured during recording.
		if e.Response.StatusCode == http.StatusBadRequest {
			continue
		}
		if e.CardID == "" {
			e.CardID = fmt.Sprintf("anon-%d", lineNum)
		}
		if e.Endpoint == "" {
			e.Endpoint = "/v1/chat/completions"
		}
		out = append(out, e)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func normalizeBaseURL(raw string) string {
	u := strings.TrimRight(strings.TrimSpace(raw), "/")
	for _, suffix := range []string{"/v1/chat/completions", "/v1/messages", "/health"} {
		if strings.HasSuffix(u, suffix) {
			return strings.TrimSuffix(u, suffix)
		}
	}
	return u
}

// ParseRetryAfter parses an HTTP Retry-After header value (seconds or HTTP-date).
func ParseRetryAfter(header string, now time.Time) (time.Duration, bool) {
	h := strings.TrimSpace(header)
	if h == "" {
		return 0, false
	}
	if secs, err := strconv.ParseInt(h, 10, 64); err == nil {
		if secs < 0 {
			secs = 0
		}
		if secs > 60 {
			secs = 60
		}
		return time.Duration(secs) * time.Second, true
	}
	if t, err := http.ParseTime(h); err == nil {
		d := t.Sub(now)
		if d < 0 {
			d = 0
		}
		if d > 60*time.Second {
			d = 60 * time.Second
		}
		return d, true
	}
	return 0, false
}

type healthTelemetry struct {
	OK                     bool   `json:"ok"`
	ActiveLanes            int    `json:"active_lanes"`
	Inflight               int    `json:"inflight"`
	PeakMemoryBytes        uint64 `json:"peak_memory_bytes"`
	MemoryBytes            uint64 `json:"memory_bytes"`
	UsedBytes              uint64 `json:"used_bytes"`
	ModelBytes             uint64 `json:"model_bytes"`
	PerLaneWorstCaseBytes  uint64 `json:"per_lane_worst_case_bytes"`
	RecommendedWorkingSetB uint64 `json:"recommended_working_set_bytes"`
}

func (h healthTelemetry) observedMemoryBytes() uint64 {
	best := h.PeakMemoryBytes
	if h.MemoryBytes > best {
		best = h.MemoryBytes
	}
	if h.UsedBytes > best {
		best = h.UsedBytes
	}
	if h.ModelBytes > 0 || h.PerLaneWorstCaseBytes > 0 {
		lanes := h.ActiveLanes
		if lanes < 0 {
			lanes = 0
		}
		est := h.ModelBytes + uint64(lanes)*h.PerLaneWorstCaseBytes
		if est > best {
			best = est
		}
	}
	return best
}

func pollHealthOnce(ctx context.Context, client *http.Client, baseURL, apiKey string, peakMem *atomic.Uint64) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/health", nil)
	if err != nil {
		return
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, resp.Body)
		return
	}
	var h healthTelemetry
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&h); err != nil {
		return
	}
	mem := h.observedMemoryBytes()
	for {
		cur := peakMem.Load()
		if mem <= cur || peakMem.CompareAndSwap(cur, mem) {
			break
		}
	}
}

func percentileMs(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	pos := p * float64(len(sorted)-1)
	lo := int(pos)
	hi := lo + 1
	if hi >= len(sorted) {
		return sorted[len(sorted)-1]
	}
	frac := pos - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

// RunSweep executes every configured concurrency rung in order and picks the knee.
func RunSweep(ctx context.Context, cfg SweepConfig, cards []Card) ([]RungMetrics, KneeResult, error) {
	if len(cards) == 0 {
		return nil, KneeResult{}, errors.New("replay set has no cards")
	}
	if len(cfg.LaneCounts) == 0 {
		return nil, KneeResult{}, errors.New("lane counts must not be empty")
	}
	results := make([]RungMetrics, 0, len(cfg.LaneCounts))
	for _, lanes := range cfg.LaneCounts {
		if ctx.Err() != nil {
			return results, KneeResult{}, ctx.Err()
		}
		if lanes < 1 {
			return results, KneeResult{}, fmt.Errorf("invalid lane count %d", lanes)
		}
		r := RunRung(ctx, cfg, lanes, cards)
		results = append(results, r)
	}
	knee, err := PickKnee(results, cfg.Knee)
	return results, knee, err
}

// RunRung drives `lanes` concurrent card workers across multi-turn cards and collects metrics.
func RunRung(ctx context.Context, cfg SweepConfig, lanes int, cards []Card) RungMetrics {
	clk := cfg.Clock
	if clk == nil {
		clk = realClock{}
	}
	client := cfg.Client
	if client == nil {
		client = &http.Client{
			Timeout: 5 * time.Minute,
			Transport: &http.Transport{
				MaxIdleConns:        lanes * 2,
				MaxIdleConnsPerHost: lanes * 2,
				IdleConnTimeout:     90 * time.Second,
			},
		}
	}
	baseURL := normalizeBaseURL(cfg.BaseURL)
	maxRetries := cfg.MaxRetries
	if maxRetries < 0 {
		maxRetries = 0
	}
	healthInterval := cfg.HealthInterval
	if healthInterval <= 0 {
		healthInterval = 250 * time.Millisecond
	}

	var peakMem atomic.Uint64
	healthCtx, cancelHealth := context.WithCancel(ctx)
	var healthWG sync.WaitGroup
	healthWG.Add(1)
	go func() {
		defer healthWG.Done()
		pollHealthOnce(healthCtx, client, baseURL, cfg.APIKey, &peakMem)
		ticker := time.NewTicker(healthInterval)
		defer ticker.Stop()
		for {
			select {
			case <-healthCtx.Done():
				return
			case <-ticker.C:
				pollHealthOnce(healthCtx, client, baseURL, cfg.APIKey, &peakMem)
			}
		}
	}()

	totalCards := cfg.CardsPerRung
	if cfg.Duration == 0 && totalCards <= 0 {
		totalCards = len(cards)
		if lanes > totalCards {
			totalCards = lanes
		}
	}

	var nextCard atomic.Int64
	var completedCards atomic.Int64
	var totalTokens atomic.Int64
	var errCount atomic.Int64
	var retry429Count atomic.Int64

	var ttftMu sync.Mutex
	var ttftsMs []float64

	start := clk.Now()
	var deadline time.Time
	if cfg.Duration > 0 {
		deadline = start.Add(cfg.Duration)
	}

	var wg sync.WaitGroup
	for w := 0; w < lanes; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				if ctx.Err() != nil {
					return
				}
				if cfg.Duration > 0 && !clk.Now().Before(deadline) {
					return
				}
				idx := int(nextCard.Add(1) - 1)
				if cfg.Duration == 0 && idx >= totalCards {
					return
				}
				template := cards[idx%len(cards)]
				replayCardID := fmt.Sprintf("%s-r%d-c%d", template.ID, lanes, idx)
				cardOK := true
				for _, turn := range template.Turns {
					toks, ttftMs, retries, err := executeTurn(ctx, client, clk, baseURL, cfg.APIKey, replayCardID, turn, maxRetries)
					retry429Count.Add(int64(retries))
					if err != nil {
						errCount.Add(1)
						cardOK = false
						break
					}
					totalTokens.Add(int64(toks))
					ttftMu.Lock()
					ttftsMs = append(ttftsMs, ttftMs)
					ttftMu.Unlock()
				}
				if cardOK {
					completedCards.Add(1)
				}
			}
		}()
	}

	wg.Wait()
	cancelHealth()
	healthWG.Wait()
	// Capture final post-rung memory snapshot if not yet sampled.
	pollHealthOnce(ctx, client, baseURL, cfg.APIKey, &peakMem)

	wall := clk.Now().Sub(start).Seconds()
	if wall <= 0 {
		wall = 1e-9
	}

	sort.Float64s(ttftsMs)
	totTok := float64(totalTokens.Load())
	doneCards := int(completedCards.Load())
	totalTokS := totTok / wall
	perLaneTokS := totalTokS / float64(lanes)

	return RungMetrics{
		Lanes:           lanes,
		TotalTokS:       totalTokS,
		PerLaneTokS:     perLaneTokS,
		TTFTP50Ms:       percentileMs(ttftsMs, 0.50),
		TTFTP95Ms:       percentileMs(ttftsMs, 0.95),
		PeakMemoryBytes: peakMem.Load(),
		CardsCompleted:  doneCards,
		CardsPerHour:    float64(doneCards) / wall * 3600.0,
		Errors:          int(errCount.Load()),
		Retries429:      int(retry429Count.Load()),
	}
}

func isStreamRequest(raw json.RawMessage) bool {
	var probe struct {
		Stream bool `json:"stream"`
	}
	_ = json.Unmarshal(raw, &probe)
	return probe.Stream
}

func executeTurn(
	ctx context.Context,
	client *http.Client,
	clk Clock,
	baseURL, apiKey, cardID string,
	turn RecordedEntry,
	maxRetries int,
) (completionTokens int, ttftMs float64, retries429 int, err error) {
	endpoint := turn.Endpoint
	if !strings.HasPrefix(endpoint, "/") {
		endpoint = "/" + endpoint
	}
	targetURL := baseURL + endpoint
	stream := isStreamRequest(turn.Request)

	for attempt := 0; ; attempt++ {
		if ctx.Err() != nil {
			return 0, 0, retries429, ctx.Err()
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(turn.Request))
		if err != nil {
			return 0, 0, retries429, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Inky-Card-Id", cardID)
		if apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+apiKey)
		}

		reqStart := clk.Now()
		resp, err := client.Do(req)
		if err != nil {
			return 0, 0, retries429, err
		}

		if resp.StatusCode == http.StatusTooManyRequests {
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if attempt >= maxRetries {
				return 0, 0, retries429, fmt.Errorf("HTTP 429 exceeded max retries (%d)", maxRetries)
			}
			retries429++
			delay, ok := ParseRetryAfter(resp.Header.Get("Retry-After"), clk.Now())
			if !ok {
				shift := attempt
				if shift > 6 {
					shift = 6
				}
				delay = (100 * time.Millisecond) << shift
				if delay > 5*time.Second {
					delay = 5 * time.Second
				}
			}
			clk.Sleep(delay)
			continue
		}

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
			resp.Body.Close()
			return 0, 0, retries429, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
		}

		if stream {
			toks, ttft, parseErr := consumeSSE(resp.Body, clk, reqStart)
			resp.Body.Close()
			return toks, ttft, retries429, parseErr
		}

		body, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		totalMs := clk.Now().Sub(reqStart).Seconds() * 1000.0
		if readErr != nil {
			return 0, 0, retries429, readErr
		}
		toks, parseErr := parseNonStreamingUsage(body)
		return toks, totalMs, retries429, parseErr
	}
}

func parseNonStreamingUsage(body []byte) (int, error) {
	var v struct {
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
			OutputTokens     int `json:"output_tokens"`
		} `json:"usage"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &v); err != nil {
		return 0, fmt.Errorf("invalid JSON response: %w", err)
	}
	if v.Error != nil {
		return 0, fmt.Errorf("server error: %s", v.Error.Message)
	}
	if v.Usage.CompletionTokens > 0 {
		return v.Usage.CompletionTokens, nil
	}
	return v.Usage.OutputTokens, nil
}

func consumeSSE(r io.Reader, clk Clock, reqStart time.Time) (int, float64, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 16*1024), 4*1024*1024)

	var firstTokMs float64
	gotFirstTok := false
	usageTokens := -1
	deltaChunks := 0
	sawTerminal := false

	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			sawTerminal = true
			continue
		}

		var ev struct {
			Type    string `json:"type"`
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Delta *struct {
				Text       string `json:"text"`
				Thinking   string `json:"thinking"`
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage *struct {
				CompletionTokens int `json:"completion_tokens"`
				OutputTokens     int `json:"output_tokens"`
			} `json:"usage"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			return 0, 0, fmt.Errorf("invalid SSE chunk: %w", err)
		}
		if ev.Error != nil {
			return 0, 0, fmt.Errorf("stream error: %s", ev.Error.Message)
		}
		if ev.Type == "message_stop" {
			sawTerminal = true
		}
		if ev.Usage != nil {
			if ev.Usage.CompletionTokens > 0 {
				usageTokens = ev.Usage.CompletionTokens
			} else if ev.Usage.OutputTokens > 0 {
				usageTokens = ev.Usage.OutputTokens
			}
		}
		hasTok := false
		if len(ev.Choices) > 0 {
			if ev.Choices[0].Delta.Content != "" || ev.Choices[0].Delta.ReasoningContent != "" {
				hasTok = true
			}
			if ev.Choices[0].FinishReason != nil && *ev.Choices[0].FinishReason != "" {
				sawTerminal = true
			}
		}
		if ev.Type == "content_block_delta" && ev.Delta != nil && (ev.Delta.Text != "" || ev.Delta.Thinking != "") {
			hasTok = true
		}
		if hasTok {
			deltaChunks++
			if !gotFirstTok {
				gotFirstTok = true
				firstTokMs = clk.Now().Sub(reqStart).Seconds() * 1000.0
			}
		}
	}
	if err := sc.Err(); err != nil {
		return 0, 0, err
	}
	if !sawTerminal {
		return 0, 0, errors.New("truncated SSE stream")
	}
	if !gotFirstTok {
		firstTokMs = clk.Now().Sub(reqStart).Seconds() * 1000.0
	}
	if usageTokens >= 0 {
		return usageTokens, firstTokMs, nil
	}
	return deltaChunks, firstTokMs, nil
}
