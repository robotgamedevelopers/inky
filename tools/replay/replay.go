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
	// Legacy recordings omit both fields. An absent generation reason also covers turns
	// that ended before the scheduler reported a completion.
	GenerationFinishReason string `json:"generation_finish_reason,omitempty"`
	DeliveryStatus         string `json:"delivery_status,omitempty"`
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

// Clock abstracts time and sleep for deterministic in-memory unit tests.
type Clock interface {
	Now() time.Time
	Sleep(d time.Duration)
}

type realClock struct{}

func (realClock) Now() time.Time        { return time.Now() }
func (realClock) Sleep(d time.Duration) { time.Sleep(d) }

// SweepConfig configures a replay lane-count sweep.
type SweepConfig struct {
	BaseURL         string
	APIKey          string
	LaneCounts      []int
	CardsPerRung    int           // 0 = max(len(cards), lanes)
	Duration        time.Duration // 0 = fixed card count mode
	MaxRetries      int
	HealthInterval  time.Duration
	ServerPID       int
	MeasureHeadroom bool
	ForceStream     bool
	RSSSampler      func(pid int) (uint64, error)
	HeadroomSampler func() (SystemMemorySample, error)
	Knee            KneeConfig
	Client          *http.Client
	Clock           Clock
}

// LoadReplaySet loads one .jsonl file or a directory of .jsonl files and their .jsonl.1 backups, groups turns
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
			if !e.IsDir() && (strings.HasSuffix(e.Name(), ".jsonl") || strings.HasSuffix(e.Name(), ".jsonl.1")) {
				files = append(files, filepath.Join(path, e.Name()))
			}
		}
		sort.Slice(files, func(i, j int) bool {
			left, right := strings.TrimSuffix(files[i], ".1"), strings.TrimSuffix(files[j], ".1")
			if left == right {
				return files[i] > files[j] // retained backup precedes the active segment
			}
			return left < right
		})
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

	var cards []Card
	grouped := make(map[string][]RecordedEntry)
	for i, e := range all {
		if e.CardID == "" {
			e.CardID = fmt.Sprintf("anon-%d", i+1)
			cards = append(cards, Card{ID: e.CardID, Turns: []RecordedEntry{e}})
			continue
		}
		grouped[e.CardID] = append(grouped[e.CardID], e)
	}

	for id, turns := range grouped {
		sort.SliceStable(turns, func(i, j int) bool {
			if turns[i].TurnIndex != turns[j].TurnIndex {
				return turns[i].TurnIndex < turns[j].TurnIndex
			}
			return turns[i].RecordedAtMs < turns[j].RecordedAtMs
		})
		// Older recorders reused card-N after restarting. When every occurrence
		// starts at turn zero there is no multi-turn sequence to reconstruct.
		_, parseErr := strconv.ParseUint(strings.TrimPrefix(id, "card-"), 10, 64)
		if strings.HasPrefix(id, "card-") && parseErr == nil && len(turns) > 1 && turns[len(turns)-1].TurnIndex == 0 {
			for _, turn := range turns {
				cards = append(cards, Card{ID: id, Turns: []RecordedEntry{turn}})
			}
		} else {
			cards = append(cards, Card{ID: id, Turns: turns})
		}
	}
	sort.SliceStable(cards, func(i, j int) bool {
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
	reader := bufio.NewReader(f)
	for lineNum := 1; ; lineNum++ {
		line, err := reader.ReadBytes('\n')
		if err != nil && err != io.EOF {
			return nil, fmt.Errorf("line %d: %w", lineNum, err)
		}
		if err == io.EOF && len(line) == 0 {
			break
		}
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var e RecordedEntry
		if err := json.Unmarshal(line, &e); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNum, err)
		}
		if len(bytes.TrimSpace(e.Request)) == 0 {
			continue
		}
		// Skip malformed 400 client requests captured during recording.
		if e.Response.StatusCode == http.StatusBadRequest {
			continue
		}
		if e.Endpoint == "" {
			e.Endpoint = "/v1/chat/completions"
		}
		out = append(out, e)
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
	ProcessFootprintBytes  uint64 `json:"process_footprint_bytes"`
	PeakMemoryBytes        uint64 `json:"peak_memory_bytes"`
	MemoryBytes            uint64 `json:"memory_bytes"`
	UsedBytes              uint64 `json:"used_bytes"`
	ModelBytes             uint64 `json:"model_bytes"`
	PerLaneWorstCaseBytes  uint64 `json:"per_lane_worst_case_bytes"`
	RecommendedWorkingSetB uint64 `json:"recommended_working_set_bytes"`
}

type memorySamples struct {
	mu              sync.Mutex
	observed        uint64
	budget          uint64
	peakRSS         uint64
	rssSamples      int
	minHeadroom     uint64
	hostMemory      uint64
	headroomSamples int
}

func (m *memorySamples) observe(h healthTelemetry) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.observed = max(m.observed, h.ProcessFootprintBytes, h.PeakMemoryBytes, h.MemoryBytes, h.UsedBytes)
	estimate := h.ModelBytes + uint64(max(0, h.ActiveLanes))*h.PerLaneWorstCaseBytes
	m.budget = max(m.budget, estimate)
}

func (m *memorySamples) observeRSS(rssBytes uint64) {
	if rssBytes == 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.peakRSS = max(m.peakRSS, rssBytes)
	m.rssSamples++
}

func (m *memorySamples) observeSystem(sys SystemMemorySample) {
	if sys.TotalBytes == 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.hostMemory = max(m.hostMemory, sys.TotalBytes)
	if m.headroomSamples == 0 || sys.AvailableBytes < m.minHeadroom {
		m.minHeadroom = sys.AvailableBytes
	}
	m.headroomSamples++
}

func (m *memorySamples) peaks() (observed, budget, peakRSS uint64, rssSamples int, minHeadroom, hostMemory uint64, headroomSamples int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.observed, m.budget, m.peakRSS, m.rssSamples, m.minHeadroom, m.hostMemory, m.headroomSamples
}

func pollTelemetryOnce(
	ctx context.Context,
	client *http.Client,
	baseURL, apiKey string,
	serverPID int,
	rssSampler func(int) (uint64, error),
	headroomSampler func() (SystemMemorySample, error),
	memory *memorySamples,
) {
	if rssSampler != nil && serverPID > 0 {
		if rss, err := rssSampler(serverPID); err == nil && rss > 0 {
			memory.observeRSS(rss)
		}
	}
	if headroomSampler != nil {
		if sys, err := headroomSampler(); err == nil && sys.TotalBytes > 0 {
			memory.observeSystem(sys)
		}
	}
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
	memory.observe(h)
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
	if cfg.Duration < 0 {
		return nil, KneeResult{}, errors.New("duration must be nonnegative")
	}
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

	rssSampler := cfg.RSSSampler
	if rssSampler == nil && cfg.ServerPID > 0 {
		rssSampler = SampleProcessRSS
	}
	headroomSampler := cfg.HeadroomSampler
	if headroomSampler == nil && cfg.MeasureHeadroom {
		headroomSampler = SampleSystemMemory
	}

	var memory memorySamples
	pollTelemetryOnce(ctx, client, baseURL, cfg.APIKey, cfg.ServerPID, rssSampler, headroomSampler, &memory)

	start := clk.Now()
	_, useRealTimers := clk.(realClock)
	healthCtx, cancelHealth := context.WithCancel(ctx)
	var healthWG sync.WaitGroup
	var nextHealthPollNs atomic.Int64
	nextHealthPollNs.Store(start.Add(healthInterval).UnixNano())
	if useRealTimers {
		healthWG.Add(1)
		go func() {
			defer healthWG.Done()
			timer := time.NewTimer(healthInterval)
			defer timer.Stop()
			for {
				select {
				case <-healthCtx.Done():
					return
				case <-timer.C:
					pollTelemetryOnce(healthCtx, client, baseURL, cfg.APIKey, cfg.ServerPID, rssSampler, headroomSampler, &memory)
					timer.Reset(healthInterval)
				}
			}
		}()
	}

	totalCards := cfg.CardsPerRung
	if cfg.Duration <= 0 && totalCards <= 0 {
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
				if cfg.Duration <= 0 && idx >= totalCards {
					return
				}
				template := cards[idx%len(cards)]
				replayCardID := fmt.Sprintf("%s-r%d-c%d", template.ID, lanes, idx)
				cardOK := true
				for _, turn := range template.Turns {
					toks, ttftMs, retries, err := executeTurnWithOptions(ctx, client, clk, baseURL, cfg.APIKey, replayCardID, turn, maxRetries, cfg.ForceStream)
					retry429Count.Add(int64(retries))
					if !useRealTimers {
						nowNs := clk.Now().UnixNano()
						targetNs := nextHealthPollNs.Load()
						if nowNs >= targetNs && nextHealthPollNs.CompareAndSwap(targetNs, nowNs+healthInterval.Nanoseconds()) {
							pollTelemetryOnce(ctx, client, baseURL, cfg.APIKey, cfg.ServerPID, rssSampler, headroomSampler, &memory)
						}
					}
					if err != nil {
						errCount.Add(1)
						cardOK = false
						break
					}
					totalTokens.Add(int64(toks))
					if ttftMs >= 0 {
						ttftMu.Lock()
						ttftsMs = append(ttftsMs, ttftMs)
						ttftMu.Unlock()
					}
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
	pollTelemetryOnce(ctx, client, baseURL, cfg.APIKey, cfg.ServerPID, rssSampler, headroomSampler, &memory)

	wall := clk.Now().Sub(start).Seconds()
	if wall <= 0 {
		wall = 1e-9
	}

	sort.Float64s(ttftsMs)
	totTok := float64(totalTokens.Load())
	doneCards := int(completedCards.Load())
	totalTokS := totTok / wall
	perLaneTokS := totalTokS / float64(lanes)

	obsPeak, budPeak, rssPeak, rssCount, minHeadroom, hostMem, headroomCount := memory.peaks()
	return RungMetrics{
		Lanes:                   lanes,
		TotalTokS:               totalTokS,
		PerLaneTokS:             perLaneTokS,
		TTFTP50Ms:               percentileMs(ttftsMs, 0.50),
		TTFTP95Ms:               percentileMs(ttftsMs, 0.95),
		TTFTSamples:             len(ttftsMs),
		PeakMemoryBytes:         obsPeak,
		PeakBudgetEstimateBytes: budPeak,
		PeakRSSBytes:            rssPeak,
		RSSSamples:              rssCount,
		MinHeadroomBytes:        minHeadroom,
		HostMemoryBytes:         hostMem,
		HeadroomSamples:         headroomCount,
		CardsCompleted:          doneCards,
		CardsPerHour:            float64(doneCards) / wall * 3600.0,
		Errors:                  int(errCount.Load()),
		Retries429:              int(retry429Count.Load()),
	}
}

func forceStreamingRequest(raw json.RawMessage) (json.RawMessage, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, err
	}
	obj["stream"] = json.RawMessage("true")
	return json.Marshal(obj)
}

func executeTurn(
	ctx context.Context,
	client *http.Client,
	clk Clock,
	baseURL, apiKey, cardID string,
	turn RecordedEntry,
	maxRetries int,
) (completionTokens int, ttftMs float64, retries429 int, err error) {
	return executeTurnWithOptions(ctx, client, clk, baseURL, apiKey, cardID, turn, maxRetries, false)
}

func executeTurnWithOptions(
	ctx context.Context,
	client *http.Client,
	clk Clock,
	baseURL, apiKey, cardID string,
	turn RecordedEntry,
	maxRetries int,
	forceStream bool,
) (completionTokens int, ttftMs float64, retries429 int, err error) {
	endpoint := turn.Endpoint
	if !strings.HasPrefix(endpoint, "/") {
		endpoint = "/" + endpoint
	}
	targetURL := baseURL + endpoint
	reqBody := turn.Request
	var options struct {
		Stream        bool `json:"stream"`
		StreamOptions struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	if err := json.Unmarshal(reqBody, &options); err != nil {
		return 0, -1, 0, fmt.Errorf("invalid recorded request: %w", err)
	}
	if forceStream && !options.Stream {
		patched, err := forceStreamingRequest(reqBody)
		if err != nil {
			return 0, -1, 0, fmt.Errorf("invalid recorded request: %w", err)
		}
		reqBody = patched
		options.Stream = true
	}

	for attempt := 0; ; attempt++ {
		if ctx.Err() != nil {
			return 0, 0, retries429, ctx.Err()
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(reqBody))
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

		messages := strings.HasSuffix(strings.TrimRight(req.URL.Path, "/"), "/v1/messages")
		if options.Stream {
			toks, ttft, parseErr := consumeSSE(resp.Body, clk, reqStart, messages, options.StreamOptions.IncludeUsage)
			resp.Body.Close()
			return toks, ttft, retries429, parseErr
		}

		body, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			return 0, 0, retries429, readErr
		}
		toks, parseErr := parseNonStreamingUsage(body, messages)
		return toks, -1, retries429, parseErr
	}
}

type replyUsage struct {
	CompletionTokens *int `json:"completion_tokens"`
	OutputTokens     *int `json:"output_tokens"`
}

func requiredTokenCount(count *int) (int, error) {
	if count == nil || *count < 0 {
		return 0, errors.New("missing or invalid completion token count")
	}
	return *count, nil
}

func parseNonStreamingUsage(body []byte, messages bool) (int, error) {
	var reply struct {
		Choices []struct {
			Message      struct{ Role string } `json:"message"`
			FinishReason string                `json:"finish_reason"`
		} `json:"choices"`
		Type       string                    `json:"type"`
		Role       string                    `json:"role"`
		Content    []json.RawMessage         `json:"content"`
		StopReason string                    `json:"stop_reason"`
		Usage      replyUsage                `json:"usage"`
		Error      *struct{ Message string } `json:"error"`
	}
	if err := json.Unmarshal(body, &reply); err != nil {
		return 0, fmt.Errorf("invalid JSON response: %w", err)
	}
	if reply.Error != nil {
		return 0, fmt.Errorf("server error: %s", reply.Error.Message)
	}
	if messages {
		if reply.Type != "message" || reply.Role != "assistant" || reply.Content == nil || reply.StopReason == "" {
			return 0, errors.New("incomplete Messages response")
		}
		return requiredTokenCount(reply.Usage.OutputTokens)
	}
	if len(reply.Choices) == 0 {
		return 0, errors.New("missing chat completion choice")
	}
	for _, choice := range reply.Choices {
		if choice.Message.Role != "assistant" || choice.FinishReason == "" {
			return 0, errors.New("incomplete chat completion choice")
		}
	}
	return requiredTokenCount(reply.Usage.CompletionTokens)
}

// The two endpoints signal completion differently. Keep terminal state separate
// from token usage so that a disconnect after a finish reason fails the turn.
type streamReply struct {
	started, finished, done bool
	blockOpen               bool
	blockIndex              int
	tokens                  *int
}

func (s *streamReply) chatChunk(payload string, separateUsage bool) (bool, error) {
	if payload == "[DONE]" {
		if !s.finished || s.tokens == nil {
			return false, errors.New("chat stream ended without finish reason or usage")
		}
		s.done = true
		return false, nil
	}
	var chunk struct {
		Choices []struct {
			Delta *struct {
				Content   string            `json:"content"`
				Reasoning string            `json:"reasoning_content"`
				ToolCalls []json.RawMessage `json:"tool_calls"`
			} `json:"delta"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *replyUsage               `json:"usage"`
		Error *struct{ Message string } `json:"error"`
	}
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		return false, fmt.Errorf("invalid chat chunk: %w", err)
	}
	if chunk.Error != nil {
		return false, fmt.Errorf("stream error: %s", chunk.Error.Message)
	}
	if chunk.Choices == nil {
		return false, errors.New("missing chat stream choices")
	}
	if len(chunk.Choices) == 0 {
		if !s.finished || chunk.Usage == nil || s.tokens != nil {
			return false, errors.New("misplaced or empty chat usage chunk")
		}
	} else {
		if len(chunk.Choices) != 1 || s.finished || chunk.Choices[0].Delta == nil {
			return false, errors.New("invalid chat stream choice")
		}
		s.finished = chunk.Choices[0].FinishReason != ""
	}
	if chunk.Usage != nil {
		if !s.finished || (separateUsage && len(chunk.Choices) != 0) {
			return false, errors.New("misplaced chat stream usage")
		}
		if _, err := requiredTokenCount(chunk.Usage.CompletionTokens); err != nil {
			return false, err
		}
		s.tokens = chunk.Usage.CompletionTokens
	}
	if len(chunk.Choices) == 0 {
		return false, nil
	}
	delta := chunk.Choices[0].Delta
	return delta.Content != "" || delta.Reasoning != "" || len(delta.ToolCalls) != 0, nil
}

func (s *streamReply) messageEvent(event, payload string) (bool, error) {
	var ev struct {
		Type    string                                `json:"type"`
		Message struct{ Type, Role string }           `json:"message"`
		Index   *int                                  `json:"index"`
		Block   struct{ Type, Text, Thinking string } `json:"content_block"`
		Delta   struct {
			Type, Text, Thinking string
			PartialJSON          string `json:"partial_json"`
			StopReason           string `json:"stop_reason"`
		} `json:"delta"`
		Usage replyUsage                `json:"usage"`
		Error *struct{ Message string } `json:"error"`
	}
	if err := json.Unmarshal([]byte(payload), &ev); err != nil {
		return false, fmt.Errorf("invalid Messages event: %w", err)
	}
	if ev.Error != nil {
		return false, fmt.Errorf("stream error: %s", ev.Error.Message)
	}
	if event != "" && event != ev.Type {
		return false, errors.New("Messages event type mismatch")
	}
	if ev.Type == "ping" {
		return false, nil
	}
	if ev.Type == "message_start" {
		if s.started || ev.Message.Type != "message" || ev.Message.Role != "assistant" {
			return false, errors.New("invalid message_start")
		}
		s.started = true
		return false, nil
	}
	if !s.started || (s.finished && ev.Type != "message_stop") {
		return false, fmt.Errorf("misplaced Messages event %q", ev.Type)
	}
	switch ev.Type {
	case "content_block_start", "content_block_delta", "content_block_stop":
		if ev.Index == nil || *ev.Index != s.blockIndex {
			return false, errors.New("unexpected Messages block index")
		}
		if ev.Type == "content_block_start" {
			if s.blockOpen || ev.Block.Type == "" {
				return false, errors.New("invalid content_block_start")
			}
			s.blockOpen = true
			return ev.Block.Type == "tool_use" || ev.Block.Text != "" || ev.Block.Thinking != "", nil
		}
		if !s.blockOpen {
			return false, errors.New("missing content_block_start")
		}
		if ev.Type == "content_block_stop" {
			s.blockOpen = false
			s.blockIndex++
			return false, nil
		}
		if ev.Delta.Type == "" {
			return false, errors.New("missing Messages block delta")
		}
		return ev.Delta.Text != "" || ev.Delta.Thinking != "" || ev.Delta.PartialJSON != "", nil
	case "message_delta":
		if s.blockOpen || ev.Delta.StopReason == "" {
			return false, errors.New("incomplete Messages stream")
		}
		if _, err := requiredTokenCount(ev.Usage.OutputTokens); err != nil {
			return false, err
		}
		s.finished = true
		s.tokens = ev.Usage.OutputTokens
	case "message_stop":
		if !s.finished {
			return false, errors.New("message_stop without final delta")
		}
		s.done = true
	default:
		return false, fmt.Errorf("unrecognized Messages event %q", ev.Type)
	}
	return false, nil
}

func consumeSSE(r io.Reader, clk Clock, reqStart time.Time, messages, separateUsage bool) (int, float64, error) {
	const maxFrame = 4 * 1024 * 1024
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 16*1024), maxFrame)
	var data strings.Builder
	var event string
	var reply streamReply
	ttft := -1.0
	for sc.Scan() {
		line := sc.Text()
		if line != "" {
			field, value, _ := strings.Cut(line, ":")
			value = strings.TrimPrefix(value, " ")
			switch field {
			case "event":
				event = value
			case "data":
				if data.Len()+len(value)+1 > maxFrame {
					return 0, -1, errors.New("SSE frame too large")
				}
				data.WriteString(value)
				data.WriteByte('\n')
			}
			continue
		}
		if data.Len() == 0 {
			event = ""
			continue
		}
		if reply.done {
			return 0, -1, errors.New("data after terminal SSE event")
		}
		payload := strings.TrimSuffix(data.String(), "\n")
		var hasOutput bool
		var err error
		if messages {
			hasOutput, err = reply.messageEvent(event, payload)
		} else {
			hasOutput, err = reply.chatChunk(payload, separateUsage)
		}
		if err != nil {
			return 0, -1, err
		}
		if hasOutput && ttft < 0 {
			ttft = clk.Now().Sub(reqStart).Seconds() * 1000
		}
		event = ""
		data.Reset()
	}
	if err := sc.Err(); err != nil {
		return 0, -1, err
	}
	if !reply.done || data.Len() != 0 {
		return 0, -1, errors.New("truncated SSE stream")
	}
	return *reply.tokens, ttft, nil
}
