// Soak and fault-storm client for `inky serve`. Reads the bearer key from
// INKY_SERVE_KEY in the environment; never prints that value or puts it on argv.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	neturl "net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
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

type soakClock interface {
	Now() time.Time
	Sleep(d time.Duration)
}

type realSoakClock struct{}

func (realSoakClock) Now() time.Time        { return time.Now() }
func (realSoakClock) Sleep(d time.Duration) { time.Sleep(d) }

type dialFunc func(network, addr string, timeout time.Duration) (net.Conn, error)

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

// withID keeps the byte length and places the request tag in the first block so
// tagged prompts do not share sealed 64-token prefix checkpoints in ServeBlockCache.
func withID(s string, id uint64) string {
	tag := fmt.Sprintf("id=%08d ", id%100000000)
	if len(tag) >= len(s) {
		return tag[:len(s)]
	}
	pos := 0
	if nl := strings.IndexByte(s, '\n'); nl >= 0 && nl+1+len(tag) <= len(s) {
		pos = nl + 1
	}
	return s[:pos] + tag + s[pos+len(tag):]
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
	Model           string     `json:"model"`
	Messages        []chatMsg  `json:"messages"`
	Tools           []toolSpec `json:"tools,omitempty"`
	ToolChoice      any        `json:"tool_choice,omitempty"`
	MaxTokens       int        `json:"max_tokens"`
	ThinkingBudget  int        `json:"thinking_budget,omitempty"`
	ReasoningEffort string     `json:"reasoning_effort,omitempty"`
	Temperature     float64    `json:"temperature"`
	Stream          bool       `json:"stream"`
	Seed            uint64     `json:"seed"`
}

type chatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type toolSpec struct {
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

type toolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
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

type healthSnapshot struct {
	OK                         bool    `json:"ok"`
	ActiveLanes                *int    `json:"active_lanes"`
	Inflight                   *int    `json:"inflight"`
	QueuedRequests             *int    `json:"queued_requests"`
	OutputTokens               *int64  `json:"output_tokens,omitempty"`
	RecommendedWorkingSetBytes *int64  `json:"recommended_working_set_bytes"`
	ModelBytes                 *int64  `json:"model_bytes"`
	PerLaneWorstCaseBytes      *int64  `json:"per_lane_worst_case_bytes"`
	MaxAdmittedLanes           *int    `json:"max_admitted_lanes"`
	StartsSpilled              *uint64 `json:"starts_spilled"`
	RecordWritten              *int64  `json:"record_written,omitempty"`
	RecordDropped              *int64  `json:"record_dropped,omitempty"`
	RecordWriteErrors          *int64  `json:"record_write_errors,omitempty"`
	RecordRotations            *int64  `json:"record_rotations,omitempty"`
}

func (h healthSnapshot) isDrained() bool {
	if !h.OK || h.ActiveLanes == nil || h.Inflight == nil {
		return false
	}
	if *h.ActiveLanes != 0 || *h.Inflight != 0 {
		return false
	}
	if h.QueuedRequests != nil && *h.QueuedRequests != 0 {
		return false
	}
	return true
}

type epochSample struct {
	Epoch          int            `json:"epoch"`
	ElapsedS       float64        `json:"elapsed_s"`
	HealthOK       bool           `json:"health_ok"`
	Health         healthSnapshot `json:"health"`
	ServerRSSBytes int64          `json:"server_rss_bytes"`
	RecordDirBytes int64          `json:"record_dir_bytes"`
	RecordDirFiles int            `json:"record_dir_files"`
	SpillDirBytes  int64          `json:"spill_dir_bytes"`
	Requests       int64          `json:"requests"`
	Completed      int64          `json:"completed"`
}

type driftConfig struct {
	MinSteadySamples    int
	WarmupEpochs        int
	MaxRSSDriftPct      float64
	MaxRSSDriftBytes    int64
	ModelBytes          int64
	RecordMaxTotalBytes int64
	SpillMaxTotalBytes  int64
}

type driftReport struct {
	OK                     bool   `json:"ok"`
	Reason                 string `json:"reason,omitempty"`
	EpochsTotal            int    `json:"epochs_total"`
	EpochsSteady           int    `json:"epochs_steady"`
	ModelBytes             int64  `json:"model_bytes"`
	RSSWarmupBytes         int64  `json:"rss_warmup_bytes"`
	RSSSteadyFirstBytes    int64  `json:"rss_steady_first_bytes"`
	RSSNetSteadyFirstBytes int64  `json:"rss_net_steady_first_bytes"`
	RSSSteadyLastBytes     int64  `json:"rss_steady_last_bytes"`
	RSSSteadyMaxBytes      int64  `json:"rss_steady_max_bytes"`
	RSSDriftBytes          int64  `json:"rss_drift_bytes"`
	RecordDirBytesMax      int64  `json:"record_dir_bytes_max"`
	SpillDirBytesMax       int64  `json:"spill_dir_bytes_max"`
}

func analyzeDrift(samples []epochSample, cfg driftConfig) driftReport {
	rep := driftReport{
		OK:          true,
		EpochsTotal: len(samples),
	}
	if len(samples) == 0 {
		rep.OK = false
		rep.Reason = "no epoch samples collected"
		return rep
	}

	var firstModelBytes int64
	for _, s := range samples {
		if s.RecordDirBytes > rep.RecordDirBytesMax {
			rep.RecordDirBytesMax = s.RecordDirBytes
		}
		if s.SpillDirBytes > rep.SpillDirBytesMax {
			rep.SpillDirBytesMax = s.SpillDirBytes
		}
		if !s.HealthOK || !s.Health.isDrained() {
			rep.OK = false
			rep.Reason = fmt.Sprintf("epoch %d did not drain cleanly (health_ok=%v active=%v inflight=%v queued=%v)",
				s.Epoch, s.HealthOK, ptrIntVal(s.Health.ActiveLanes), ptrIntVal(s.Health.Inflight), ptrIntVal(s.Health.QueuedRequests))
			return rep
		}
		if cfg.RecordMaxTotalBytes > 0 && s.RecordDirBytes > cfg.RecordMaxTotalBytes {
			rep.OK = false
			rep.Reason = fmt.Sprintf("epoch %d record_dir_bytes=%d exceeded cap=%d", s.Epoch, s.RecordDirBytes, cfg.RecordMaxTotalBytes)
			return rep
		}
		if cfg.SpillMaxTotalBytes > 0 && s.SpillDirBytes > cfg.SpillMaxTotalBytes {
			rep.OK = false
			rep.Reason = fmt.Sprintf("epoch %d spill_dir_bytes=%d exceeded cap=%d", s.Epoch, s.SpillDirBytes, cfg.SpillMaxTotalBytes)
			return rep
		}
		if s.Health.ModelBytes != nil && *s.Health.ModelBytes > 0 {
			if firstModelBytes == 0 {
				firstModelBytes = *s.Health.ModelBytes
			} else if *s.Health.ModelBytes != firstModelBytes {
				rep.OK = false
				rep.Reason = fmt.Sprintf("epoch %d model_bytes changed: got=%d want=%d", s.Epoch, *s.Health.ModelBytes, firstModelBytes)
				return rep
			}
			if s.Health.RecommendedWorkingSetBytes != nil && *s.Health.RecommendedWorkingSetBytes <= *s.Health.ModelBytes {
				rep.OK = false
				rep.Reason = fmt.Sprintf("epoch %d recommended_working_set_bytes (%d) <= model_bytes (%d)",
					s.Epoch, *s.Health.RecommendedWorkingSetBytes, *s.Health.ModelBytes)
				return rep
			}
		}
	}

	warmup := cfg.WarmupEpochs
	if warmup < 0 {
		warmup = 0
	}
	if warmup > 0 && len(samples) >= warmup {
		rep.RSSWarmupBytes = samples[warmup-1].ServerRSSBytes
	}
	steadyStart := warmup
	if steadyStart > len(samples) {
		steadyStart = len(samples)
	}
	steady := samples[steadyStart:]
	rep.EpochsSteady = len(steady)
	if len(steady) < cfg.MinSteadySamples {
		rep.OK = false
		rep.Reason = fmt.Sprintf("collected %d steady epochs, need %d", len(steady), cfg.MinSteadySamples)
		return rep
	}
	if len(steady) == 0 {
		return rep
	}

	rep.RSSSteadyFirstBytes = steady[0].ServerRSSBytes
	rep.RSSSteadyLastBytes = steady[len(steady)-1].ServerRSSBytes
	rep.RSSSteadyMaxBytes = steady[0].ServerRSSBytes
	for _, s := range steady[1:] {
		if s.ServerRSSBytes > rep.RSSSteadyMaxBytes {
			rep.RSSSteadyMaxBytes = s.ServerRSSBytes
		}
	}
	effectiveModelBytes := cfg.ModelBytes
	if effectiveModelBytes <= 0 {
		effectiveModelBytes = firstModelBytes
	}
	rep.ModelBytes = effectiveModelBytes
	netFirst := rep.RSSSteadyFirstBytes
	if effectiveModelBytes > 0 {
		if rep.RSSSteadyFirstBytes > effectiveModelBytes {
			netFirst = rep.RSSSteadyFirstBytes - effectiveModelBytes
		} else {
			netFirst = 0
		}
	}
	rep.RSSNetSteadyFirstBytes = netFirst

	if rep.RSSSteadyFirstBytes > 0 {
		rep.RSSDriftBytes = rep.RSSSteadyLastBytes - rep.RSSSteadyFirstBytes
		if len(steady) >= 2 {
			// Measure percentage drift net of static model weights so multi-GiB models cannot mask heap/KV leaks.
			pctAllowed := int64(float64(netFirst) * (cfg.MaxRSSDriftPct / 100.0))
			allowed := cfg.MaxRSSDriftBytes
			if pctAllowed > allowed {
				allowed = pctAllowed
			}
			if rep.RSSDriftBytes > allowed {
				rep.OK = false
				rep.Reason = fmt.Sprintf("steady-state RSS drifted upward by %d bytes net of model=%d (first=%d net_first=%d last=%d allowed=%d)",
					rep.RSSDriftBytes, effectiveModelBytes, rep.RSSSteadyFirstBytes, netFirst, rep.RSSSteadyLastBytes, allowed)
				return rep
			}
			// Detect sustained upward RSS creep across 4+ steady epochs even when an intermediate epoch is flat or dips slightly.
			if len(steady) >= 4 && cfg.MaxRSSDriftBytes > 0 && sustainedUpwardRSSCreep(steady, cfg.MaxRSSDriftBytes/2) {
				rep.OK = false
				rep.Reason = fmt.Sprintf("steady-state RSS crept upward across %d epochs (%d -> %d, delta=%d)",
					len(steady), rep.RSSSteadyFirstBytes, rep.RSSSteadyLastBytes, rep.RSSDriftBytes)
				return rep
			}
		}
	}

	// If spill directory is monitored across 3+ steady epochs, ensure post-drain spill bytes do not climb.
	if len(steady) >= 3 && sustainedUpwardSpillCreep(steady) {
		rep.OK = false
		rep.Reason = fmt.Sprintf("steady-state spill_dir_bytes crept upward (%d -> %d)",
			steady[0].SpillDirBytes, steady[len(steady)-1].SpillDirBytes)
		return rep
	}
	return rep
}

func sustainedUpwardRSSCreep(s []epochSample, threshold int64) bool {
	n := len(s)
	if n < 4 {
		return false
	}
	netDelta := s[n-1].ServerRSSBytes - s[0].ServerRSSBytes
	if netDelta <= threshold {
		return false
	}
	nonDecreasing := true
	upSteps := 0
	for i := 1; i < n; i++ {
		if s[i].ServerRSSBytes < s[i-1].ServerRSSBytes {
			nonDecreasing = false
		} else if s[i].ServerRSSBytes > s[i-1].ServerRSSBytes {
			upSteps++
		}
	}
	if nonDecreasing && upSteps >= 2 {
		return true
	}
	// Linear regression trend check catches steady leaks even if one noisy epoch dips slightly.
	if upSteps*2 <= n-1 {
		return false
	}
	meanX := float64(n-1) / 2.0
	var sumY float64
	for _, ep := range s {
		sumY += float64(ep.ServerRSSBytes)
	}
	meanY := sumY / float64(n)
	var covXY, varX, varY float64
	for i, ep := range s {
		dx := float64(i) - meanX
		dy := float64(ep.ServerRSSBytes) - meanY
		covXY += dx * dy
		varX += dx * dx
		varY += dy * dy
	}
	if varX <= 0 || varY <= 0 {
		return false
	}
	slope := covXY / varX
	r2 := (covXY * covXY) / (varX * varY)
	return slope > 0 && slope*float64(n-1) > float64(threshold) && r2 >= 0.70
}

func sustainedUpwardSpillCreep(s []epochSample) bool {
	n := len(s)
	if n < 3 || s[n-1].SpillDirBytes <= s[0].SpillDirBytes {
		return false
	}
	upSteps := 0
	for i := 1; i < n; i++ {
		if s[i].SpillDirBytes < s[i-1].SpillDirBytes {
			return false
		}
		if s[i].SpillDirBytes > s[i-1].SpillDirBytes {
			upSteps++
		}
	}
	return upSteps >= 2
}

func unexpected429UnderCapacity(clients, lanes, queue int, allow429 bool, count429 int64) bool {
	return !allow429 && clients <= lanes+queue && count429 > 0
}

func ptrIntVal(p *int) int {
	if p == nil {
		return -1
	}
	return *p
}

func parseFlexibleDuration(raw string) (time.Duration, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return 0, errors.New("empty duration")
	}
	if secs, err := strconv.ParseInt(s, 10, 64); err == nil {
		if secs < 1 {
			return 0, fmt.Errorf("duration must be >= 1s: %s", raw)
		}
		return time.Duration(secs) * time.Second, nil
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return 0, err
	}
	if d < time.Second {
		return 0, fmt.Errorf("duration must be >= 1s: %s", raw)
	}
	return d, nil
}

func chooseEpochDuration(total time.Duration, requested time.Duration) time.Duration {
	if requested > 0 {
		if requested > total {
			return total
		}
		return requested
	}
	if total < 4*time.Second {
		return total
	}
	ep := total / 6
	if ep < 5*time.Second {
		ep = 5 * time.Second
	}
	if ep > 5*time.Minute {
		ep = 5 * time.Minute
	}
	if ep > total {
		return total
	}
	return ep
}

func sampleProcessRSS(pid int) (int64, error) {
	if pid <= 0 {
		return 0, nil
	}
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		return 0, err
	}
	kb, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return 0, err
	}
	return kb * 1024, nil
}

func measureDirBytes(dir string) (int64, int, error) {
	if dir == "" {
		return 0, 0, nil
	}
	var total int64
	var count int
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			if path != dir && os.IsNotExist(walkErr) {
				return nil
			}
			return walkErr
		}
		if path == dir && !d.IsDir() {
			return fmt.Errorf("%s is not a directory", dir)
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		total += info.Size()
		count++
		return nil
	})
	return total, count, err
}

func newSoakHTTPClient() *http.Client {
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
	return &http.Client{
		Transport: transport,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

type clientConfig struct {
	chatURL               string
	key                   string
	duration              time.Duration
	epoch                 time.Duration
	minEpochs             int
	warmupEpochs          int
	maxTokensChoices      []int
	clients               int
	lanes                 int
	queue                 int
	allow429              bool
	seed                  uint64
	disconnectPct         int
	serverPID             int
	modelBytes            int64
	recordDir             string
	recordMaxTotalBytes   int64
	spillDir              string
	spillMaxTotalBytes    int64
	maxRSSDriftPct        float64
	maxRSSDriftBytes      int64
	mode                  string
	stormRequests         int
	expectRecordWriteErrs bool
	referenceFile         string
	clock                 soakClock
	dial                  dialFunc
	stdout                io.Writer
	stderr                io.Writer
}

func (c clientConfig) clk() soakClock {
	if c.clock != nil {
		return c.clock
	}
	return realSoakClock{}
}

func (c clientConfig) dialConn(network, addr string, timeout time.Duration) (net.Conn, error) {
	if c.dial != nil {
		return c.dial(network, addr, timeout)
	}
	return net.DialTimeout(network, addr, timeout)
}

func (c clientConfig) outWriter() io.Writer {
	if c.stdout != nil {
		return c.stdout
	}
	return os.Stdout
}

func (c clientConfig) errWriter() io.Writer {
	if c.stderr != nil {
		return c.stderr
	}
	return os.Stderr
}

func formatTokens(tokens []int) string {
	if len(tokens) == 0 {
		tokens = maxTokensChoices
	}
	values := make([]string, len(tokens))
	for i, v := range tokens {
		values[i] = strconv.Itoa(v)
	}
	return strings.Join(values, ",")
}

func main() {
	os.Exit(runCLI(os.Args[1:], os.Getenv, nil, clientConfig{}))
}

func runCLI(args []string, getenv func(string) string, httpClient *http.Client, base clientConfig) int {
	out := base.outWriter()
	errOut := base.errWriter()
	fs := flag.NewFlagSet("soak", flag.ContinueOnError)
	fs.SetOutput(errOut)
	urlFlag := fs.String("url", "http://127.0.0.1:18090/v1/chat/completions", "chat completions URL")
	durationFlag := fs.String("duration", "3600", "run duration (seconds or Go duration like 60s, 72h)")
	epochFlag := fs.String("epoch", "0", "longitudinal sampling epoch duration (0 = auto)")
	minEpochs := fs.Int("min-epochs", 0, "optional minimum number of epochs to complete")
	warmupEpochs := fs.Int("warmup-epochs", 1, "warmup epochs before steady-state drift analysis")
	maxTokensFlag := fs.String("max-tokens", "", "optional comma-separated max_tokens choices (default 16,64,256)")
	clients := fs.Int("clients", 24, "concurrent clients")
	lanes := fs.Int("lanes", 16, "server lane count, recorded in the summary")
	queue := fs.Int("queue", 64, "server queue depth, recorded in the summary")
	allow429 := fs.Bool("allow-429", false, "allow HTTP 429 responses even when clients <= lanes + queue")
	seed := fs.Uint64("seed", 1, "RNG seed")
	disconnect := fs.Int("disconnect-pct", 30, "percent of requests closed mid-response")
	serverPID := fs.Int("server-pid", 0, "optional server PID for RSS drift sampling")
	modelBytes := fs.Int64("model-bytes", 0, "optional model weight bytes override for net RSS drift analysis")
	recordDir := fs.String("record-dir", "", "optional server --record-dir to verify bounded disk usage")
	recordMaxTotalBytes := fs.Int64("record-max-total-bytes", 0, "optional max total bytes allowed in record-dir")
	spillDir := fs.String("spill-dir", "", "optional spill directory to verify bounded disk usage")
	spillMaxTotalBytes := fs.Int64("spill-max-total-bytes", 0, "optional max total bytes allowed in spill-dir")
	maxRSSDriftPct := fs.Float64("max-rss-drift-pct", 15.0, "allowed steady-state RSS drift percentage")
	maxRSSDriftBytes := fs.Int64("max-rss-drift-bytes", 64*1024*1024, "minimum allowed steady-state RSS drift bytes")
	mode := fs.String("mode", "soak", "mode: soak | disconnect-storm | fault-check | reference-check")
	stormRequests := fs.Int("storm-requests", 72, "total overlapping requests in disconnect-storm mode")
	expectRecordWriteErrs := fs.Bool("expect-record-write-errors", false, "require /health record_write_errors > 0 in fault-check mode")
	referenceFile := fs.String("reference-file", "", "path to save/compare seeded solo reference in reference-check mode")
	selfCheck := fs.Bool("self-check", false, "check prompt lengths and exit")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	parsedMaxTokens := maxTokensChoices
	if strings.TrimSpace(*maxTokensFlag) != "" {
		parsedMaxTokens = nil
		for _, part := range strings.Split(*maxTokensFlag, ",") {
			v, convErr := strconv.Atoi(strings.TrimSpace(part))
			if convErr != nil || v < 1 {
				fmt.Fprintf(errOut, "bad -max-tokens: %s\n", *maxTokensFlag)
				return 2
			}
			parsedMaxTokens = append(parsedMaxTokens, v)
		}
	}

	if *selfCheck {
		for i, k := range promptKinds {
			p := prompt(i, k.n)
			if len(p) != k.n {
				fmt.Fprintf(errOut, "prompt %s length %d want %d\n", k.name, len(p), k.n)
				return 1
			}
			q := withID(p, 42)
			if len(q) != k.n {
				fmt.Fprintf(errOut, "prompt %s id length %d want %d\n", k.name, len(q), k.n)
				return 1
			}
		}
		fmt.Fprintf(out, "prompt_short_bytes=%d\n", lenShort)
		fmt.Fprintf(out, "prompt_medium_bytes=%d\n", lenMedium)
		fmt.Fprintf(out, "prompt_long_bytes=%d\n", lenLong)
		fmt.Fprintf(out, "max_tokens=%s\n", formatTokens(parsedMaxTokens))
		fmt.Fprintln(out, "self_check=ok")
		return 0
	}

	dur, err := parseFlexibleDuration(*durationFlag)
	if err != nil {
		fmt.Fprintf(errOut, "bad -duration: %v\n", err)
		return 2
	}
	var epDur time.Duration
	if strings.TrimSpace(*epochFlag) != "" && strings.TrimSpace(*epochFlag) != "0" && strings.TrimSpace(*epochFlag) != "0s" {
		epDur, err = parseFlexibleDuration(*epochFlag)
		if err != nil {
			fmt.Fprintf(errOut, "bad -epoch: %v\n", err)
			return 2
		}
	}
	if *clients < 1 || *lanes < 1 || *queue < 0 || *disconnect < 0 || *disconnect > 100 || *stormRequests < 1 || *minEpochs < 0 {
		fmt.Fprintln(errOut, "clients, lanes, queue, storm-requests, min-epochs, or disconnect-pct is out of range")
		return 2
	}
	key := getenv("INKY_SERVE_KEY")
	if key == "" {
		fmt.Fprintln(errOut, "INKY_SERVE_KEY is not set")
		return 2
	}

	cfg := clientConfig{
		chatURL:               *urlFlag,
		key:                   key,
		duration:              dur,
		epoch:                 epDur,
		minEpochs:             *minEpochs,
		warmupEpochs:          *warmupEpochs,
		maxTokensChoices:      parsedMaxTokens,
		clients:               *clients,
		lanes:                 *lanes,
		queue:                 *queue,
		allow429:              *allow429,
		seed:                  *seed,
		disconnectPct:         *disconnect,
		serverPID:             *serverPID,
		modelBytes:            *modelBytes,
		recordDir:             *recordDir,
		recordMaxTotalBytes:   *recordMaxTotalBytes,
		spillDir:              *spillDir,
		spillMaxTotalBytes:    *spillMaxTotalBytes,
		maxRSSDriftPct:        *maxRSSDriftPct,
		maxRSSDriftBytes:      *maxRSSDriftBytes,
		mode:                  *mode,
		stormRequests:         *stormRequests,
		expectRecordWriteErrs: *expectRecordWriteErrs,
		referenceFile:         *referenceFile,
		clock:                 base.clock,
		dial:                  base.dial,
		stdout:                out,
		stderr:                errOut,
	}

	if httpClient == nil {
		httpClient = newSoakHTTPClient()
	}
	rssFn := sampleProcessRSS
	if cfg.clock != nil && cfg.serverPID == 0 {
		rssFn = func(int) (int64, error) { return 0, nil }
	}
	switch cfg.mode {
	case "soak":
		return runSoak(httpClient, cfg, rssFn)
	case "disconnect-storm":
		return runDisconnectStorm(httpClient, cfg)
	case "fault-check":
		return runFaultCheck(httpClient, cfg)
	case "reference-check":
		return runReferenceCheck(httpClient, cfg)
	default:
		fmt.Fprintf(errOut, "unknown -mode: %s\n", cfg.mode)
		return 2
	}
}

func runSoak(httpClient *http.Client, cfg clientConfig, rssSampler func(int) (int64, error)) int {
	clk := cfg.clk()
	out := cfg.outWriter()
	errOut := cfg.errWriter()
	var tot totals
	startTime := clk.Now()
	totalDeadline := startTime.Add(cfg.duration)
	epochDur := chooseEpochDuration(cfg.duration, cfg.epoch)
	hURL := healthURL(cfg.chatURL)

	var initialSpills uint64
	minSteady := 0
	if cfg.serverPID > 0 || cfg.recordDir != "" {
		minSteady = 2
	}
	if cfg.spillDir != "" {
		minSteady = 3 // the disk growth check compares three drained snapshots
		initial, ok := waitHealthDrainWithClock(httpClient, hURL, 5*time.Second, clk)
		if !ok || !initial.isDrained() || initial.StartsSpilled == nil {
			fmt.Fprintln(errOut, "missing initial spill counter on drained server")
			return 1
		}
		initialSpills = *initial.StartsSpilled
	}
	var samples []epochSample
	epochIdx := 0

	for epochIdx == 0 || clk.Now().Before(totalDeadline) || epochIdx < cfg.minEpochs {
		epochDeadline := clk.Now().Add(epochDur)
		if epochIdx >= cfg.minEpochs && epochDeadline.After(totalDeadline) {
			epochDeadline = totalDeadline
		}

		ctxEpoch, cancelEpoch := context.WithCancel(context.Background())
		var wg sync.WaitGroup
		if cfg.clock == nil {
			go func(idx int) {
				tick := time.NewTicker(30 * time.Second)
				defer tick.Stop()
				for {
					select {
					case <-ctxEpoch.Done():
						return
					case <-tick.C:
						fmt.Fprintf(errOut, "soak tick epoch=%d requests=%d inflight=%d 5xx=%d disconnected=%d\n",
							idx, tot.requests.Load(), tot.inflight.Load(), tot.http5xx.Load(), tot.disconnected.Load())
					}
				}
			}(epochIdx)
		}

		for w := 0; w < cfg.clients; w++ {
			wg.Add(1)
			go func(w int, ep int) {
				defer wg.Done()
				rng := rand.New(rand.NewPCG(cfg.seed^uint64(ep*7919), uint64(w)+1))
				for clk.Now().Before(epochDeadline) && ctxEpoch.Err() == nil {
					oneRequest(&tot, httpClient, cfg.key, cfg.chatURL, cfg.disconnectPct, cfg.seed, "", cfg.maxTokensChoices, clk, rng)
				}
			}(w, epochIdx)
		}
		wg.Wait()
		cancelEpoch()

		snap, healthOK := waitHealthDrainWithClock(httpClient, hURL, 10*time.Second, clk)
		rssBytes, rssErr := rssSampler(cfg.serverPID)
		if cfg.serverPID > 0 && (rssErr != nil || rssBytes <= 0) {
			fmt.Fprintf(errOut, "epoch %d RSS probe for pid %d: bytes=%d err=%v\n", epochIdx, cfg.serverPID, rssBytes, rssErr)
			return 1
		}
		recBytes, recFiles, recErr := measureDirBytes(cfg.recordDir)
		spillBytes, _, spillErr := measureDirBytes(cfg.spillDir)
		if recErr != nil || spillErr != nil {
			fmt.Fprintf(errOut, "epoch %d directory measurement failed: record=%v spill=%v\n", epochIdx, recErr, spillErr)
			return 1
		}

		s := epochSample{
			Epoch:          epochIdx,
			ElapsedS:       clk.Now().Sub(startTime).Seconds(),
			HealthOK:       healthOK,
			Health:         snap,
			ServerRSSBytes: rssBytes,
			RecordDirBytes: recBytes,
			RecordDirFiles: recFiles,
			SpillDirBytes:  spillBytes,
			Requests:       tot.requests.Load(),
			Completed:      tot.completed.Load(),
		}
		samples = append(samples, s)
		fmt.Fprintf(errOut, "soak epoch=%d elapsed=%.1fs active_lanes=%d inflight=%d queued=%d rss_bytes=%d record_bytes=%d record_files=%d\n",
			s.Epoch, s.ElapsedS, ptrIntVal(s.Health.ActiveLanes), ptrIntVal(s.Health.Inflight), ptrIntVal(s.Health.QueuedRequests),
			s.ServerRSSBytes, s.RecordDirBytes, s.RecordDirFiles)
		epochIdx++
	}

	if cfg.spillDir != "" {
		last := samples[len(samples)-1].Health.StartsSpilled
		if last == nil || *last <= initialSpills {
			fmt.Fprintln(errOut, "no successful spill observed during the workload")
			return 1
		}
		fmt.Fprintf(out, "starts_spilled=%d\n", *last-initialSpills)
	}
	inflightEnd := tot.inflight.Load()
	probeOK := 0
	if inflightEnd == 0 {
		probeOK = runRecoveryProbes(httpClient, cfg.key, cfg.chatURL, cfg.lanes, cfg.seed)
	}
	finalSnap, serverHealthOK := waitHealthDrainWithClock(httpClient, hURL, 5*time.Second, clk)
	serverActiveLanes := ptrIntVal(finalSnap.ActiveLanes)
	serverInflight := ptrIntVal(finalSnap.Inflight)
	serverQueued := 0
	if finalSnap.QueuedRequests != nil {
		serverQueued = *finalSnap.QueuedRequests
	}

	drift := analyzeDrift(samples, driftConfig{
		MinSteadySamples:    minSteady,
		WarmupEpochs:        cfg.warmupEpochs,
		MaxRSSDriftPct:      cfg.maxRSSDriftPct,
		MaxRSSDriftBytes:    cfg.maxRSSDriftBytes,
		ModelBytes:          cfg.modelBytes,
		RecordMaxTotalBytes: cfg.recordMaxTotalBytes,
		SpillMaxTotalBytes:  cfg.spillMaxTotalBytes,
	})
	bad429 := unexpected429UnderCapacity(cfg.clients, cfg.lanes, cfg.queue, cfg.allow429, tot.http429.Load())

	fmt.Fprintf(out, "seed=%d\n", cfg.seed)
	fmt.Fprintf(out, "duration_s=%d\n", int(cfg.duration.Seconds()))
	fmt.Fprintf(out, "clients=%d\n", cfg.clients)
	fmt.Fprintf(out, "lanes=%d\n", cfg.lanes)
	fmt.Fprintf(out, "queue=%d\n", cfg.queue)
	fmt.Fprintf(out, "disconnect_pct=%d\n", cfg.disconnectPct)
	fmt.Fprintf(out, "prompt_short_bytes=%d\n", lenShort)
	fmt.Fprintf(out, "prompt_medium_bytes=%d\n", lenMedium)
	fmt.Fprintf(out, "prompt_long_bytes=%d\n", lenLong)
	fmt.Fprintf(out, "max_tokens=%s\n", formatTokens(cfg.maxTokensChoices))
	fmt.Fprintf(out, "requests=%d\n", tot.requests.Load())
	fmt.Fprintf(out, "http_200=%d\n", tot.http200.Load())
	fmt.Fprintf(out, "http_4xx=%d\n", tot.http4xx.Load())
	fmt.Fprintf(out, "http_401=%d\n", tot.http401.Load())
	fmt.Fprintf(out, "http_429=%d\n", tot.http429.Load())
	fmt.Fprintf(out, "http_5xx=%d\n", tot.http5xx.Load())
	fmt.Fprintf(out, "disconnected=%d\n", tot.disconnected.Load())
	fmt.Fprintf(out, "transport_errors=%d\n", tot.transport.Load())
	fmt.Fprintf(out, "timeouts=%d\n", tot.timeouts.Load())
	fmt.Fprintf(out, "inflight_end=%d\n", inflightEnd)
	fmt.Fprintf(out, "inflight_max=%d\n", tot.inflightMax.Load())
	fmt.Fprintf(out, "server_active_lanes_end=%d\n", serverActiveLanes)
	fmt.Fprintf(out, "server_inflight_end=%d\n", serverInflight)
	fmt.Fprintf(out, "server_queued_end=%d\n", serverQueued)
	fmt.Fprintf(out, "server_health_ok=%d\n", boolInt(serverHealthOK))
	fmt.Fprintf(out, "workload_completed=%d\n", tot.completed.Load())
	fmt.Fprintf(out, "unexpected_status=%d\n", tot.unexpected.Load())
	fmt.Fprintf(out, "stream_errors=%d\n", tot.streamErrors.Load())
	fmt.Fprintf(out, "recovery_completed=%d\n", probeOK)
	fmt.Fprintf(out, "recovery_requested=%d\n", cfg.lanes)
	fmt.Fprintf(out, "epochs_total=%d\n", drift.EpochsTotal)
	fmt.Fprintf(out, "epochs_steady=%d\n", drift.EpochsSteady)
	fmt.Fprintf(out, "model_bytes=%d\n", drift.ModelBytes)
	fmt.Fprintf(out, "rss_warmup_bytes=%d\n", drift.RSSWarmupBytes)
	fmt.Fprintf(out, "rss_steady_first_bytes=%d\n", drift.RSSSteadyFirstBytes)
	fmt.Fprintf(out, "rss_net_steady_first_bytes=%d\n", drift.RSSNetSteadyFirstBytes)
	fmt.Fprintf(out, "rss_steady_last_bytes=%d\n", drift.RSSSteadyLastBytes)
	fmt.Fprintf(out, "rss_steady_max_bytes=%d\n", drift.RSSSteadyMaxBytes)
	fmt.Fprintf(out, "rss_drift_bytes=%d\n", drift.RSSDriftBytes)
	fmt.Fprintf(out, "record_dir_bytes_max=%d\n", drift.RecordDirBytesMax)
	fmt.Fprintf(out, "spill_dir_bytes_max=%d\n", drift.SpillDirBytesMax)
	fmt.Fprintf(out, "unexpected_429=%d\n", boolInt(bad429))
	fmt.Fprintf(out, "drift_ok=%d\n", boolInt(drift.OK))
	if !drift.OK {
		fmt.Fprintf(out, "drift_reason=%s\n", drift.Reason)
	}
	fmt.Fprintln(out, "inflight_source=server_health_and_client_outstanding")

	if inflightEnd != 0 || !serverHealthOK || serverActiveLanes != 0 || serverInflight != 0 || serverQueued != 0 ||
		tot.http5xx.Load() != 0 || tot.http401.Load() != 0 || bad429 || tot.unexpected.Load() != 0 || tot.streamErrors.Load() != 0 ||
		tot.completed.Load() == 0 || tot.transport.Load() != 0 || tot.timeouts.Load() != 0 || probeOK != cfg.lanes || !drift.OK {
		return 1
	}
	return 0
}

func runRecoveryProbes(httpClient *http.Client, key, chatURL string, lanes int, seed uint64) int {
	var probes sync.WaitGroup
	var ok atomic.Int64
	start := make(chan struct{})
	for i := 0; i < lanes; i++ {
		probes.Add(1)
		go func(i int) {
			defer probes.Done()
			<-start
			if probe(httpClient, key, chatURL, seed+uint64(i)) {
				ok.Add(1)
			}
		}(i)
	}
	close(start)
	probes.Wait()
	return int(ok.Load())
}

type seededReply struct {
	Content          string `json:"content"`
	FinishReason     string `json:"finish_reason"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
}

func runSeededSoloReference(httpClient *http.Client, key, chatURL string, seed uint64) (seededReply, error) {
	payload, err := json.Marshal(chatReq{
		Model:           "inkling-2-mini",
		Messages:        []chatMsg{{Role: "user", Content: "What is 17 times 23? Answer with only the integer."}},
		MaxTokens:       32,
		ReasoningEffort: "none",
		Temperature:     0,
		Stream:          false,
		Seed:            seed,
	})
	if err != nil {
		return seededReply{}, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, chatURL, bytes.NewReader(payload))
	if err != nil {
		return seededReply{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	req.Close = true
	resp, err := httpClient.Do(req)
	if err != nil {
		return seededReply{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return seededReply{}, fmt.Errorf("status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var out struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return seededReply{}, err
	}
	if len(out.Choices) == 0 {
		return seededReply{}, errors.New("empty choices in seeded reply")
	}
	return seededReply{
		Content:          out.Choices[0].Message.Content,
		FinishReason:     out.Choices[0].FinishReason,
		PromptTokens:     out.Usage.PromptTokens,
		CompletionTokens: out.Usage.CompletionTokens,
	}, nil
}

func runReferenceCheck(httpClient *http.Client, cfg clientConfig) int {
	clk := cfg.clk()
	out := cfg.outWriter()
	errOut := cfg.errWriter()
	refSeed := cfg.seed ^ 0x9001
	var baseline *seededReply
	if cfg.referenceFile != "" {
		if data, err := os.ReadFile(cfg.referenceFile); err == nil {
			var r seededReply
			if err := json.Unmarshal(data, &r); err != nil {
				fmt.Fprintf(errOut, "invalid reference file %s: %v\n", cfg.referenceFile, err)
				return 1
			}
			baseline = &r
		}
	}

	got, err := runSeededSoloReference(httpClient, cfg.key, cfg.chatURL, refSeed)
	if err != nil {
		fmt.Fprintf(errOut, "seeded solo reference failed: %v\n", err)
		return 1
	}
	if baseline == nil {
		if cfg.referenceFile != "" {
			data, _ := json.Marshal(got)
			if err := os.WriteFile(cfg.referenceFile, data, 0o600); err != nil {
				fmt.Fprintf(errOut, "write reference file: %v\n", err)
				return 1
			}
		}
		fmt.Fprintf(out, "reference_saved=1 finish_reason=%s completion_tokens=%d\n", got.FinishReason, got.CompletionTokens)
		return 0
	}

	if got != *baseline {
		fmt.Fprintf(errOut, "seeded reference mismatch: got=%+v want=%+v\n", got, *baseline)
		return 1
	}
	// Also verify multi-lane concurrent execution matches exact seeded output.
	var wg sync.WaitGroup
	var mismatch atomic.Int64
	for i := 0; i < cfg.lanes; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := runSeededSoloReference(httpClient, cfg.key, cfg.chatURL, refSeed)
			if err != nil || r != *baseline {
				mismatch.Add(1)
			}
		}()
	}
	wg.Wait()
	snap, healthOK := waitHealthDrainWithClock(httpClient, healthURL(cfg.chatURL), 5*time.Second, clk)
	if mismatch.Load() != 0 || !healthOK || !snap.isDrained() {
		fmt.Fprintf(errOut, "post-check mismatch=%d health_ok=%v drained=%v\n", mismatch.Load(), healthOK, snap.isDrained())
		return 1
	}
	fmt.Fprintf(out, "reference_match=1 lanes_verified=%d finish_reason=%s completion_tokens=%d\n",
		cfg.lanes, got.FinishReason, got.CompletionTokens)
	return 0
}

type disconnectStage int

const (
	stageZeroMs disconnectStage = iota
	stageHeaderDone
	stageTTFT
	stageMidThinking
	stageMidDecode
	stageComplete
)

func (s disconnectStage) String() string {
	switch s {
	case stageZeroMs:
		return "0ms"
	case stageHeaderDone:
		return "header-done"
	case stageTTFT:
		return "ttft"
	case stageMidThinking:
		return "mid-thinking"
	case stageMidDecode:
		return "mid-decode"
	default:
		return "complete"
	}
}

func verifyEarlyAbortsStopGeneration(stormTot *totals, httpClient *http.Client, cfg clientConfig) (int64, int64, int64, bool) {
	clk := cfg.clk()
	hURL := healthURL(cfg.chatURL)
	snap0, ok0 := waitHealthDrainWithClock(httpClient, hURL, 10*time.Second, clk)
	if !ok0 || !snap0.isDrained() || snap0.OutputTokens == nil {
		return -1, -1, 0, false
	}

	var preTot totals
	rng := rand.New(rand.NewPCG(cfg.seed^0x7711, 1))
	nPre := 4
	for i := 0; i < nPre; i++ {
		runStormOne(&preTot, httpClient, cfg, stageZeroMs, cfg.seed^uint64(1000+i), 256, rng)
		runStormOne(&preTot, httpClient, cfg, stageHeaderDone, cfg.seed^uint64(2000+i), 256, rng)
	}
	stormTot.transport.Add(preTot.transport.Load())
	stormTot.timeouts.Add(preTot.timeouts.Load())
	snap1, ok1 := waitHealthDrainWithClock(httpClient, hURL, 10*time.Second, clk)
	if !ok1 || !snap1.isDrained() || snap1.OutputTokens == nil {
		return -1, -1, 0, false
	}
	preTTFTGen := *snap1.OutputTokens - *snap0.OutputTokens

	var ttftTot totals
	nTTFT := cfg.lanes
	if nTTFT < 2 {
		nTTFT = 2
	}
	if nTTFT > 4 {
		nTTFT = 4
	}
	const ttftMaxTokens = 256
	var wg sync.WaitGroup
	for i := 0; i < nTTFT; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			r := rand.New(rand.NewPCG(cfg.seed^0x8822, uint64(i)+1))
			runStormOne(&ttftTot, httpClient, cfg, stageTTFT, cfg.seed^uint64(3000+i), ttftMaxTokens, r)
		}(i)
	}
	wg.Wait()
	stormTot.transport.Add(ttftTot.transport.Load())
	stormTot.http5xx.Add(ttftTot.http5xx.Load())
	stormTot.http401.Add(ttftTot.http401.Load())
	stormTot.unexpected.Add(ttftTot.unexpected.Load())
	stormTot.streamErrors.Add(ttftTot.streamErrors.Load())
	stormTot.timeouts.Add(ttftTot.timeouts.Load())
	snap2, ok2 := waitHealthDrainWithClock(httpClient, hURL, 10*time.Second, clk)
	if !ok2 || !snap2.isDrained() || snap2.OutputTokens == nil {
		return preTTFTGen, -1, 0, false
	}
	ttftGen := *snap2.OutputTokens - *snap1.OutputTokens
	ttft200 := ttftTot.http200.Load()
	maxBudget := ttft200 * ttftMaxTokens
	noFaults := preTot.transport.Load() == 0 && ttftTot.transport.Load() == 0 &&
		ttftTot.http5xx.Load() == 0 && ttftTot.http401.Load() == 0 &&
		ttftTot.unexpected.Load() == 0 && ttftTot.streamErrors.Load() == 0 &&
		ttftTot.timeouts.Load() == 0
	// 0ms and header-done aborts happen before a valid body completes and must generate 0 tokens;
	// ttft aborts close immediately after the first SSE byte and must stop far short of max_tokens=256.
	ok := noFaults && preTTFTGen == 0 && ttft200 > 0 && ttftGen >= 0 && ttftGen < ttft200*64
	return preTTFTGen, ttftGen, maxBudget, ok
}

func runDisconnectStorm(httpClient *http.Client, cfg clientConfig) int {
	clk := cfg.clk()
	out := cfg.outWriter()
	errOut := cfg.errWriter()
	refSeed := cfg.seed ^ 0x9001
	baseline, err := runSeededSoloReference(httpClient, cfg.key, cfg.chatURL, refSeed)
	if err != nil {
		fmt.Fprintf(errOut, "pre-storm solo reference failed: %v\n", err)
		return 1
	}

	var tot totals
	preTTFTGen, ttftGen, ttftBudget, earlyAbortOK := verifyEarlyAbortsStopGeneration(&tot, httpClient, cfg)
	var stageCounts [6]atomic.Int64
	var nextIdx atomic.Int64
	totalReqs := int64(cfg.stormRequests)
	if totalReqs < 64 {
		totalReqs = 64
	}

	var wg sync.WaitGroup
	for w := 0; w < cfg.clients; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(cfg.seed^0x5150, uint64(w)+1))
			for {
				idx := nextIdx.Add(1) - 1
				if idx >= totalReqs {
					return
				}
				stage := disconnectStage(idx % 5)
				if idx%10 == 9 {
					stage = stageComplete
				}
				if runStormOne(&tot, httpClient, cfg, stage, cfg.seed^uint64(idx), 0, rng) {
					stageCounts[stage].Add(1)
				}
			}
		}(w)
	}
	wg.Wait()

	hURL := healthURL(cfg.chatURL)
	snap, healthOK := waitHealthDrainWithClock(httpClient, hURL, 20*time.Second, clk)
	drained := healthOK && snap.isDrained()

	postRef, err := runSeededSoloReference(httpClient, cfg.key, cfg.chatURL, refSeed)
	refMatch := err == nil && postRef == baseline

	probeOK := 0
	if drained {
		probeOK = runRecoveryProbes(httpClient, cfg.key, cfg.chatURL, cfg.lanes, cfg.seed)
	}
	bad429 := unexpected429UnderCapacity(cfg.clients, cfg.lanes, cfg.queue, cfg.allow429, tot.http429.Load())

	fmt.Fprintf(out, "storm_requests=%d\n", tot.requests.Load())
	fmt.Fprintf(out, "storm_stage_0ms=%d\n", stageCounts[stageZeroMs].Load())
	fmt.Fprintf(out, "storm_stage_header_done=%d\n", stageCounts[stageHeaderDone].Load())
	fmt.Fprintf(out, "storm_stage_ttft=%d\n", stageCounts[stageTTFT].Load())
	fmt.Fprintf(out, "storm_stage_mid_thinking=%d\n", stageCounts[stageMidThinking].Load())
	fmt.Fprintf(out, "storm_stage_mid_decode=%d\n", stageCounts[stageMidDecode].Load())
	fmt.Fprintf(out, "storm_stage_complete=%d\n", stageCounts[stageComplete].Load())
	fmt.Fprintf(out, "storm_http_200=%d\n", tot.http200.Load())
	fmt.Fprintf(out, "storm_http_429=%d\n", tot.http429.Load())
	fmt.Fprintf(out, "storm_http_5xx=%d\n", tot.http5xx.Load())
	fmt.Fprintf(out, "storm_disconnected=%d\n", tot.disconnected.Load())
	fmt.Fprintf(out, "storm_completed=%d\n", tot.completed.Load())
	fmt.Fprintf(out, "storm_transport_errors=%d\n", tot.transport.Load())
	fmt.Fprintf(out, "storm_stream_errors=%d\n", tot.streamErrors.Load())
	fmt.Fprintf(out, "storm_early_pre_ttft_tokens=%d\n", preTTFTGen)
	fmt.Fprintf(out, "storm_early_ttft_tokens=%d\n", ttftGen)
	fmt.Fprintf(out, "storm_early_ttft_max_budget=%d\n", ttftBudget)
	fmt.Fprintf(out, "early_abort_stopped=%d\n", boolInt(earlyAbortOK))
	fmt.Fprintf(out, "unexpected_429=%d\n", boolInt(bad429))
	fmt.Fprintf(out, "drain_ok=%d\n", boolInt(drained))
	fmt.Fprintf(out, "server_active_lanes_end=%d\n", ptrIntVal(snap.ActiveLanes))
	fmt.Fprintf(out, "server_inflight_end=%d\n", ptrIntVal(snap.Inflight))
	fmt.Fprintf(out, "server_queued_end=%d\n", ptrIntVal(snap.QueuedRequests))
	fmt.Fprintf(out, "ref_match=%d\n", boolInt(refMatch))
	fmt.Fprintf(out, "recovery_completed=%d\n", probeOK)
	fmt.Fprintf(out, "recovery_requested=%d\n", cfg.lanes)

	for i := range stageCounts {
		if stageCounts[i].Load() == 0 {
			fmt.Fprintf(errOut, "storm did not reach %s\n", disconnectStage(i))
			return 1
		}
	}
	if tot.http5xx.Load() != 0 || tot.http401.Load() != 0 || bad429 || tot.unexpected.Load() != 0 ||
		tot.transport.Load() != 0 || tot.streamErrors.Load() != 0 || tot.timeouts.Load() != 0 ||
		!earlyAbortOK || !drained || !refMatch || probeOK != cfg.lanes {
		return 1
	}
	return 0
}

func runStormOne(tot *totals, httpClient *http.Client, cfg clientConfig, stage disconnectStage, reqSeed uint64, overrideMaxTokens int, rng *rand.Rand) bool {
	clk := cfg.clk()
	key := cfg.key
	chatURL := cfg.chatURL
	tot.enter()
	defer tot.inflight.Add(-1)
	id := uint64(tot.requests.Add(1))

	u, err := neturl.Parse(chatURL)
	if err != nil || u.Host == "" {
		tot.transport.Add(1)
		return false
	}

	if stage == stageZeroMs {
		c, err := cfg.dialConn("tcp", u.Host, 5*time.Second)
		if err != nil {
			tot.transport.Add(1)
			return false
		}
		_ = c.Close()
		tot.disconnected.Add(1)
		return true
	}

	payload := buildStormPayload(stage, id, reqSeed, overrideMaxTokens, rng)
	if stage == stageHeaderDone {
		c, err := cfg.dialConn("tcp", u.Host, 5*time.Second)
		if err != nil {
			tot.transport.Add(1)
			return false
		}
		partialLen := len(payload) / 2
		if partialLen < 1 {
			partialLen = 1
		}
		hdr := fmt.Sprintf("POST %s HTTP/1.1\r\nHost: %s\r\nAuthorization: Bearer %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n",
			u.RequestURI(), u.Host, key, len(payload))
		if _, err := c.Write(append([]byte(hdr), payload[:partialLen]...)); err != nil {
			_ = c.Close()
			tot.transport.Add(1)
			return false
		}
		clk.Sleep(time.Duration(1+rng.IntN(5)) * time.Millisecond)
		_ = c.Close()
		tot.disconnected.Add(1)
		return true
	}

	box := &dialBox{}
	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
	defer cancel()
	ctx = context.WithValue(ctx, ctxKey{}, box)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, chatURL, bytes.NewReader(payload))
	if err != nil {
		tot.transport.Add(1)
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("X-Inky-Card-Id", fmt.Sprintf("storm-card-%d", id%16))
	req.Close = true

	resp, err := httpClient.Do(req)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			tot.timeouts.Add(1)
		} else {
			tot.transport.Add(1)
		}
		return false
	}
	defer resp.Body.Close()
	tot.note(resp.StatusCode)
	if resp.StatusCode != http.StatusOK {
		return false
	}

	reached, err := readStreamUntil(resp.Body, stage)
	if err != nil {
		fmt.Fprintf(os.Stderr, "storm stream error (id=%d stage=%s): %v\n", id, stage, err)
		if ctx.Err() == context.DeadlineExceeded {
			tot.timeouts.Add(1)
		} else {
			tot.streamErrors.Add(1)
		}
		return false
	}
	if reached {
		if stage == stageComplete {
			tot.completed.Add(1)
		} else {
			if c := box.conn(); c != nil {
				_ = c.Close()
			}
			tot.disconnected.Add(1)
		}
		if stage == stageMidThinking {
			clk.Sleep(time.Duration(1+rng.IntN(25)) * time.Millisecond)
		} else if stage == stageMidDecode {
			clk.Sleep(time.Duration(15+rng.IntN(160)) * time.Millisecond)
		}
	}
	return reached
}

func buildStormPayload(stage disconnectStage, id, reqSeed uint64, overrideMaxTokens int, rng *rand.Rand) []byte {
	var r chatReq
	switch {
	case stage == stageMidDecode:
		r = chatReq{
			Model:     "inkling-2-mini",
			Messages:  []chatMsg{{Role: "user", Content: fmt.Sprintf("Write a paragraph about the ocean (trial %d).", id)}},
			MaxTokens: 128, ReasoningEffort: "none", Stream: true, Seed: reqSeed,
		}
	case stage == stageMidThinking || id%3 == 1:
		r = chatReq{
			Model:           "inkling-2-mini",
			Messages:        []chatMsg{{Role: "user", Content: fmt.Sprintf("Solve step by step: find small positive integer solutions to x^2 - 7y^2 = 1 (trial %d).", id)}},
			MaxTokens:       256,
			ThinkingBudget:  192,
			ReasoningEffort: "high",
			Temperature:     0.7,
			Stream:          true,
			Seed:            reqSeed,
		}
	case id%3 == 2:
		r = chatReq{
			Model: "inkling-2-mini",
			Messages: []chatMsg{
				{Role: "user", Content: fmt.Sprintf("Call get_weather for Seattle and Tokyo (trial %d).", id)},
			},
			Tools: []toolSpec{{
				Type: "function",
				Function: toolFunction{
					Name:        "get_weather",
					Description: "Get weather for a city",
					Parameters: map[string]any{
						"type":       "object",
						"properties": map[string]any{"city": map[string]any{"type": "string"}},
						"required":   []string{"city"},
					},
				},
			}},
			ToolChoice:  "auto",
			MaxTokens:   128,
			Temperature: 0,
			Stream:      true,
			Seed:        reqSeed,
		}
	default:
		r = chatReq{
			Model:       "inkling-2-mini",
			Messages:    []chatMsg{{Role: "user", Content: withID(prompt(int(rng.IntN(2)), promptKinds[rng.IntN(2)].n), id)}},
			MaxTokens:   128,
			Temperature: 0,
			Stream:      true,
			Seed:        reqSeed,
		}
	}
	if overrideMaxTokens > 0 {
		r.MaxTokens = overrideMaxTokens
	}
	b, _ := json.Marshal(r)
	return b
}

func runFaultCheck(httpClient *http.Client, cfg clientConfig) int {
	clk := cfg.clk()
	out := cfg.outWriter()
	errOut := cfg.errWriter()
	var tot totals
	deadline := clk.Now().Add(cfg.duration)
	var wg sync.WaitGroup
	for w := 0; w < cfg.clients; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(cfg.seed^0xfa11, uint64(w)+1))
			for clk.Now().Before(deadline) {
				oneWithCard(&tot, httpClient, cfg.key, cfg.chatURL, cfg.seed, fmt.Sprintf("fault-card-%d", w), clk, rng)
			}
		}(w)
	}
	wg.Wait()

	snap, healthOK := waitHealthDrainWithClock(httpClient, healthURL(cfg.chatURL), 10*time.Second, clk)
	drained := healthOK && snap.isDrained()
	var writeErrs int64
	if snap.RecordWriteErrors != nil {
		writeErrs = *snap.RecordWriteErrors
	}
	var recWritten int64
	if snap.RecordWritten != nil {
		recWritten = *snap.RecordWritten
	}
	bad429 := unexpected429UnderCapacity(cfg.clients, cfg.lanes, cfg.queue, cfg.allow429, tot.http429.Load())

	fmt.Fprintf(out, "requests=%d\n", tot.requests.Load())
	fmt.Fprintf(out, "http_200=%d\n", tot.http200.Load())
	fmt.Fprintf(out, "http_429=%d\n", tot.http429.Load())
	fmt.Fprintf(out, "http_5xx=%d\n", tot.http5xx.Load())
	fmt.Fprintf(out, "workload_completed=%d\n", tot.completed.Load())
	fmt.Fprintf(out, "stream_errors=%d\n", tot.streamErrors.Load())
	fmt.Fprintf(out, "transport_errors=%d\n", tot.transport.Load())
	fmt.Fprintf(out, "timeouts=%d\n", tot.timeouts.Load())
	fmt.Fprintf(out, "unexpected_429=%d\n", boolInt(bad429))
	fmt.Fprintf(out, "drain_ok=%d\n", boolInt(drained))
	fmt.Fprintf(out, "record_written=%d\n", recWritten)
	fmt.Fprintf(out, "record_write_errors=%d\n", writeErrs)

	if tot.inflight.Load() != 0 || !drained || tot.http5xx.Load() != 0 || tot.http401.Load() != 0 || bad429 ||
		tot.unexpected.Load() != 0 || tot.streamErrors.Load() != 0 || tot.completed.Load() == 0 ||
		tot.transport.Load() != 0 || tot.timeouts.Load() != 0 {
		return 1
	}
	if cfg.expectRecordWriteErrs && writeErrs == 0 {
		fmt.Fprintln(errOut, "expected record_write_errors > 0 under faulted --record-dir, got 0")
		return 1
	}
	return 0
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

func waitHealthDrain(httpClient *http.Client, url string, timeout time.Duration) (healthSnapshot, bool) {
	return waitHealthDrainWithClock(httpClient, url, timeout, realSoakClock{})
}

func waitHealthDrainWithClock(httpClient *http.Client, url string, timeout time.Duration, clk soakClock) (healthSnapshot, bool) {
	deadline := clk.Now().Add(timeout)
	var last healthSnapshot
	var lastOK bool
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
					last = snap
					lastOK = true
					cancel()
					if snap.isDrained() {
						return last, true
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
		if clk.Now().After(deadline) {
			return last, lastOK
		}
		clk.Sleep(25 * time.Millisecond)
	}
}

func oneWithCard(tot *totals, httpClient *http.Client, key, url string, seed uint64, cardID string, clk soakClock, rng *rand.Rand) {
	oneRequest(tot, httpClient, key, url, 0, seed, cardID, maxTokensChoices, clk, rng)
}

func oneRequest(tot *totals, httpClient *http.Client, key, url string, disconnectPct int, seed uint64, cardID string, tokChoices []int, clk soakClock, rng *rand.Rand) {
	tot.enter()
	defer tot.inflight.Add(-1)
	id := uint64(tot.requests.Add(1))

	kind := int(rng.UintN(uint(len(promptKinds))))
	bodyText := prompt(kind, promptKinds[kind].n)
	if rng.IntN(2) == 0 {
		bodyText = withID(bodyText, id)
	}
	if len(tokChoices) == 0 {
		tokChoices = maxTokensChoices
	}
	maxTok := tokChoices[rng.IntN(len(tokChoices))]
	disconnect := disconnectPct > 0 && rng.IntN(100) < disconnectPct
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
	if cardID != "" {
		req.Header.Set("X-Inky-Card-Id", cardID)
	}
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
		clk.Sleep(hold)
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
	_, err := readStreamUntil(r, stageComplete)
	return err
}

func readStreamUntil(r io.Reader, stage disconnectStage) (bool, error) {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 4096), 1024*1024)
	finished, done := false, false
	var dataLines []string
	visibleEvents := 0
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
			return false, fmt.Errorf("data after DONE")
		}
		if data == "[DONE]" {
			done = true
			continue
		}
		var event struct {
			Error   json.RawMessage `json:"error"`
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					Reasoning string `json:"reasoning_content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return false, err
		}
		if len(event.Error) > 0 && string(event.Error) != "null" {
			return false, fmt.Errorf("stream error")
		}
		for _, choice := range event.Choices {
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				finished = true
			}
			if finished {
				continue
			}
			visible := choice.Delta.Content != ""
			thinking := choice.Delta.Reasoning != ""
			if visible {
				visibleEvents++
			}
			switch stage {
			case stageTTFT:
				if visible || thinking {
					return true, nil
				}
			case stageMidThinking:
				if thinking {
					return true, nil
				}
			case stageMidDecode:
				if visibleEvents >= 2 {
					return true, nil
				}
			}
		}
	}
	if len(dataLines) > 0 {
		return false, fmt.Errorf("unterminated SSE event")
	}
	if err := s.Err(); err != nil {
		return false, err
	}
	if !finished || !done {
		return false, fmt.Errorf("incomplete stream")
	}
	return stage == stageComplete, nil
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
