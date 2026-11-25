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
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Clock abstracts time for deterministic unit tests.
type Clock interface {
	Now() time.Time
	Sleep(d time.Duration)
}

type realClock struct{}

func (realClock) Now() time.Time        { return time.Now() }
func (realClock) Sleep(d time.Duration) { time.Sleep(d) }

// RunConfig configures one quality replay pass against an OpenAI-compatible inky serve instance.
type RunConfig struct {
	Name            string
	BaseURL         string
	APIKey          string
	Model           string
	KVCacheMode     string
	ReasoningEffort string
	Temperature     float64
	Seed            int64
	MaxTokens       int
	MaxRetries      int
	Client          *http.Client
	Clock           Clock
	Progress        io.Writer
}

// LoadCards reads benchmark cards from a JSONL file.
func LoadCards(path string) ([]Card, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var cards []Card
	seen := make(map[string]bool)
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
		var c Card
		if err := json.Unmarshal(line, &c); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNum, err)
		}
		if strings.TrimSpace(c.ID) == "" {
			return nil, fmt.Errorf("line %d: card id is required", lineNum)
		}
		if seen[c.ID] {
			return nil, fmt.Errorf("line %d: duplicate card id %q", lineNum, c.ID)
		}
		seen[c.ID] = true
		switch c.Kind {
		case KindMechanicalEdit, KindBugfix, KindMultiStepAlgorithm:
		default:
			return nil, fmt.Errorf("line %d: invalid card kind %q", lineNum, c.Kind)
		}
		if len(c.Messages) == 0 {
			return nil, fmt.Errorf("line %d: card %q has no messages", lineNum, c.ID)
		}
		cards = append(cards, c)
	}
	if len(cards) == 0 {
		return nil, fmt.Errorf("no cards loaded from %s", path)
	}
	return cards, nil
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

// FetchHealth queries GET /health on the target server.
func FetchHealth(ctx context.Context, client *http.Client, baseURL, apiKey string) (HealthInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, normalizeBaseURL(baseURL)+"/health", nil)
	if err != nil {
		return HealthInfo{}, err
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return HealthInfo{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return HealthInfo{}, fmt.Errorf("GET /health HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var h HealthInfo
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&h); err != nil {
		return HealthInfo{}, err
	}
	return h, nil
}

// RunSuite executes all benchmark cards against one server configuration and collects deterministic metrics.
func RunSuite(ctx context.Context, cfg RunConfig, cards []Card) (ModeRun, error) {
	if len(cards) == 0 {
		return ModeRun{}, errors.New("cards must not be empty")
	}
	clk := cfg.Clock
	if clk == nil {
		clk = realClock{}
	}
	client := cfg.Client
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Minute}
	}
	baseURL := normalizeBaseURL(cfg.BaseURL)
	health, err := FetchHealth(ctx, client, baseURL, cfg.APIKey)
	if err != nil {
		return ModeRun{}, err
	}
	kvMode := cfg.KVCacheMode
	if kvMode == "" {
		if health.KV8 {
			kvMode = "kv8"
		} else {
			kvMode = "fp16"
		}
	}
	model := cfg.Model
	if model == "" {
		model = "inkling-2-mini"
	}

	run := ModeRun{
		Name:            cfg.Name,
		KVCacheMode:     kvMode,
		ReasoningEffort: cfg.ReasoningEffort,
		Temperature:     cfg.Temperature,
		Seed:            cfg.Seed,
		Health:          health,
		CardsTotal:      len(cards),
		Results:         make([]CardResult, 0, len(cards)),
	}
	if run.Name == "" {
		run.Name = fmt.Sprintf("%s-%s", kvMode, cfg.ReasoningEffort)
	}

	var sumWall, sumDecode float64
	for i, card := range cards {
		chatRes, execErr := executeCard(ctx, client, clk, baseURL, cfg, model, card, i)
		var cardRes CardResult
		if execErr != nil {
			cardRes = CardResult{
				CardID:   card.ID,
				Kind:     card.Kind,
				Passed:   false,
				Error:    execErr.Error(),
				Response: chatRes,
			}
		} else {
			cardRes = EvaluateCard(card, chatRes)
		}
		if cardRes.Passed {
			run.CardsPassed++
		}
		if cfg.Progress != nil {
			fmt.Fprintf(cfg.Progress, "[%s] %s (%s): pass=%v (%d/%d) comp_toks=%d wall=%.0fms\n",
				run.Name, card.ID, card.Kind, cardRes.Passed, cardRes.ChecksPassed, cardRes.ChecksTotal,
				cardRes.Response.CompletionTokens, cardRes.Response.WallMs)
		}
		sumWall += cardRes.Response.WallMs
		sumDecode += float64(cardRes.Response.DecodeMs)
		run.TotalCompTokens += cardRes.Response.CompletionTokens
		run.Results = append(run.Results, cardRes)
	}
	run.TotalWallMs = sumWall
	run.PassRate = float64(run.CardsPassed) / float64(run.CardsTotal)
	run.AvgWallMs = sumWall / float64(run.CardsTotal)
	run.AvgDecodeMs = sumDecode / float64(run.CardsTotal)
	run.AvgCompTokens = float64(run.TotalCompTokens) / float64(run.CardsTotal)
	return run, nil
}

func executeCard(
	ctx context.Context,
	client *http.Client,
	clk Clock,
	baseURL string,
	cfg RunConfig,
	model string,
	card Card,
	idx int,
) (ChatResult, error) {
	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = card.MaxTokens
	}
	if maxTokens <= 0 {
		maxTokens = 512
	}
	seed := cfg.Seed + int64(idx)
	payload := map[string]any{
		"model":       model,
		"messages":    card.Messages,
		"max_tokens":  maxTokens,
		"temperature": cfg.Temperature,
		"seed":        seed,
		"stream":      false,
	}
	if cfg.ReasoningEffort != "" {
		payload["reasoning_effort"] = cfg.ReasoningEffort
	}
	if len(card.Tools) > 0 {
		payload["tools"] = card.Tools
	}
	reqBytes, err := json.Marshal(payload)
	if err != nil {
		return ChatResult{}, err
	}

	maxRetries := cfg.MaxRetries
	if maxRetries < 0 {
		maxRetries = 0
	}
	for attempt := 0; ; attempt++ {
		if ctx.Err() != nil {
			return ChatResult{}, ctx.Err()
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(reqBytes))
		if err != nil {
			return ChatResult{}, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Inky-Card-Id", fmt.Sprintf("quality-%s-%s", cfg.Name, card.ID))
		if cfg.APIKey != "" {
			req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
		}

		start := clk.Now()
		resp, err := client.Do(req)
		if err != nil {
			return ChatResult{}, err
		}

		if resp.StatusCode == http.StatusTooManyRequests {
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if attempt >= maxRetries {
				return ChatResult{}, fmt.Errorf("HTTP 429 exceeded max retries (%d)", maxRetries)
			}
			delay := (100 * time.Millisecond) << min(attempt, 6)
			if secs, err := strconv.Atoi(strings.TrimSpace(resp.Header.Get("Retry-After"))); err == nil && secs > 0 {
				delay = time.Duration(min(secs, 60)) * time.Second
			}
			clk.Sleep(delay)
			continue
		}

		body, readErr := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		resp.Body.Close()
		wallMs := float64(clk.Now().Sub(start).Microseconds()) / 1000.0
		if readErr != nil {
			return ChatResult{WallMs: wallMs}, readErr
		}
		if resp.StatusCode != http.StatusOK {
			return ChatResult{WallMs: wallMs}, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
		}

		var parsed struct {
			Choices []struct {
				Message struct {
					Content   string     `json:"content"`
					Reasoning string     `json:"reasoning_content"`
					ToolCalls []ToolCall `json:"tool_calls"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage struct {
				PromptTokens        int `json:"prompt_tokens"`
				CompletionTokens    int `json:"completion_tokens"`
				PromptTokensDetails struct {
					CachedTokens int `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(body, &parsed); err != nil {
			return ChatResult{WallMs: wallMs}, fmt.Errorf("decode response: %w", err)
		}
		if len(parsed.Choices) == 0 {
			return ChatResult{WallMs: wallMs}, errors.New("response contained 0 choices")
		}
		ch := parsed.Choices[0]
		qMs, _ := strconv.ParseInt(resp.Header.Get("X-Inky-Queue-Ms"), 10, 64)
		pMs, _ := strconv.ParseInt(resp.Header.Get("X-Inky-Prefill-Ms"), 10, 64)
		dMs, _ := strconv.ParseInt(resp.Header.Get("X-Inky-Decode-Ms"), 10, 64)
		return ChatResult{
			Content:          ch.Message.Content,
			Reasoning:        ch.Message.Reasoning,
			ToolCalls:        ch.Message.ToolCalls,
			FinishReason:     ch.FinishReason,
			PromptTokens:     parsed.Usage.PromptTokens,
			CompletionTokens: parsed.Usage.CompletionTokens,
			CachedTokens:     parsed.Usage.PromptTokensDetails.CachedTokens,
			WallMs:           wallMs,
			QueueMs:          qMs,
			PrefillMs:        pMs,
			DecodeMs:         dMs,
		}, nil
	}
}

var lexicalTokenRe = regexp.MustCompile(`[A-Za-z_][A-Za-z0-9_]*|[0-9]+|\S`)

// LexicalTokens splits text into deterministic code/prose lexical tokens for agreement scoring.
func LexicalTokens(s string) []string {
	return lexicalTokenRe.FindAllString(s, -1)
}

// TokenAgreement computes positional token agreement and longest-common-prefix ratio between two outputs.
func TokenAgreement(ref, cand string) (exact bool, positionalRatio float64, prefixRatio float64) {
	if ref == cand {
		return true, 1.0, 1.0
	}
	rt := LexicalTokens(ref)
	ct := LexicalTokens(cand)
	maxLen := max(len(rt), len(ct))
	if maxLen == 0 {
		return true, 1.0, 1.0
	}
	minLen := min(len(rt), len(ct))
	posMatches := 0
	prefixMatches := 0
	prefixBroken := false
	for i := 0; i < minLen; i++ {
		if rt[i] == ct[i] {
			posMatches++
			if !prefixBroken {
				prefixMatches++
			}
		} else {
			prefixBroken = true
		}
	}
	return false, float64(posMatches) / float64(maxLen), float64(prefixMatches) / float64(maxLen)
}

// CompareModes computes quality, temp=0 token/output agreement, KV memory savings, and latency deltas.
func CompareModes(ref, cand ModeRun) (ModeComparison, error) {
	if len(ref.Results) == 0 {
		return ModeComparison{}, errors.New("reference run has no card results")
	}
	candByID := make(map[string]CardResult, len(cand.Results))
	for _, r := range cand.Results {
		candByID[r.CardID] = r
	}

	comp := ModeComparison{
		ReferenceName:       ref.Name,
		CandidateName:       cand.Name,
		CardsTotal:          len(ref.Results),
		RefPassed:           ref.CardsPassed,
		CandPassed:          cand.CardsPassed,
		RefPassRate:         ref.PassRate,
		CandPassRate:        cand.PassRate,
		RefKVBytesPerElem:   ref.Health.KVBytesPerElem,
		CandKVBytesPerElem:  cand.Health.KVBytesPerElem,
		RefMaxAdmittedLanes: ref.Health.MaxAdmittedLanes,
		CandMaxAdmittedLanes: cand.Health.MaxAdmittedLanes,
		RefAvgWallMs:        ref.AvgWallMs,
		CandAvgWallMs:       cand.AvgWallMs,
		RefAvgDecodeMs:      ref.AvgDecodeMs,
		CandAvgDecodeMs:     cand.AvgDecodeMs,
	}
	if comp.RefKVBytesPerElem <= 0 {
		comp.RefKVBytesPerElem = 2
	}
	if comp.CandKVBytesPerElem <= 0 {
		comp.CandKVBytesPerElem = 1
	}
	comp.KVElementReductionRatio = float64(comp.RefKVBytesPerElem) / float64(comp.CandKVBytesPerElem)
	const gib = 1024.0 * 1024.0 * 1024.0
	comp.RefPerLaneWorstCaseGiB = float64(ref.Health.PerLaneWorstCaseBytes) / gib
	comp.CandPerLaneWorstCaseGiB = float64(cand.Health.PerLaneWorstCaseBytes) / gib
	if cand.Health.PerLaneWorstCaseBytes > 0 {
		comp.PerLaneMemorySavingRatio = float64(ref.Health.PerLaneWorstCaseBytes) / float64(cand.Health.PerLaneWorstCaseBytes)
	}

	var exactCount, parityCount int
	var sumTokAgree, sumPrefixAgree float64
	for _, rr := range ref.Results {
		cr, ok := candByID[rr.CardID]
		if !ok {
			return ModeComparison{}, fmt.Errorf("candidate run missing card %q", rr.CardID)
		}
		exact, tokRatio, prefixRatio := TokenAgreement(rr.NormalizedOutput, cr.NormalizedOutput)
		if exact {
			exactCount++
		}
		if rr.Passed == cr.Passed {
			parityCount++
		}
		sumTokAgree += tokRatio
		sumPrefixAgree += prefixRatio
		comp.Cards = append(comp.Cards, CardAgreement{
			CardID:              rr.CardID,
			Kind:                rr.Kind,
			RefPassed:           rr.Passed,
			CandPassed:          cr.Passed,
			ExactMatch:          exact,
			TokenAgreementRatio: tokRatio,
			PrefixMatchRatio:    prefixRatio,
			RefCompTokens:       rr.Response.CompletionTokens,
			CandCompTokens:      cr.Response.CompletionTokens,
			RefWallMs:           rr.Response.WallMs,
			CandWallMs:          cr.Response.WallMs,
		})
	}
	n := float64(len(ref.Results))
	comp.CheckParityRate = float64(parityCount) / n
	comp.ExactOutputAgreement = float64(exactCount) / n
	comp.MeanTokenAgreement = sumTokAgree / n
	comp.MeanPrefixMatch = sumPrefixAgree / n
	return comp, nil
}

// BuildEffortSweep summarizes pass rate, tokens, and latency per (effort, kind) and identifies
// the lowest thinking effort achieving 100% pass rate per card kind.
func BuildEffortSweep(runs []ModeRun, effortOrder []string) EffortSweepReport {
	if len(effortOrder) == 0 {
		effortOrder = []string{"none", "low", "medium", "high"}
	}
	kinds := []string{KindMechanicalEdit, KindBugfix, KindMultiStepAlgorithm}
	byEffort := make(map[string]ModeRun, len(runs))
	for _, r := range runs {
		byEffort[r.ReasoningEffort] = r
	}

	var cells []EffortKindMetrics
	cellLookup := make(map[string]EffortKindMetrics)
	for _, eff := range effortOrder {
		run, ok := byEffort[eff]
		if !ok {
			continue
		}
		for _, kind := range kinds {
			var total, passed, tokens int
			var wall float64
			for _, res := range run.Results {
				if res.Kind != kind {
					continue
				}
				total++
				if res.Passed {
					passed++
				}
				tokens += res.Response.CompletionTokens
				wall += res.Response.WallMs
			}
			if total == 0 {
				continue
			}
			m := EffortKindMetrics{
				ReasoningEffort: eff,
				Kind:            kind,
				CardsTotal:      total,
				CardsPassed:     passed,
				PassRate:        float64(passed) / float64(total),
				TotalCompTokens: tokens,
				AvgCompTokens:   float64(tokens) / float64(total),
				TotalWallMs:     wall,
				AvgWallMs:       wall / float64(total),
			}
			cells = append(cells, m)
			cellLookup[eff+"|"+kind] = m
		}
	}

	var recs []KindEffortRecommendation
	for _, kind := range kinds {
		rec := KindEffortRecommendation{Kind: kind}
		for _, eff := range effortOrder {
			if m, ok := cellLookup[eff+"|"+kind]; ok && m.CardsTotal > 0 && m.CardsPassed == m.CardsTotal {
				rec.LowestPassingEffort = eff
				rec.AvgTokensAtLowest = m.AvgCompTokens
				rec.AvgWallMsAtLowest = m.AvgWallMs
				break
			}
		}
		recs = append(recs, rec)
	}

	return EffortSweepReport{
		Efforts:         effortOrder,
		Cells:           cells,
		Recommendations: recs,
	}
}

// FormatMarkdown renders a concise, deterministic Markdown summary of KV comparison and effort sweeps.
func FormatMarkdown(report FullReport) string {
	var b strings.Builder
	if report.Comparison != nil {
		c := report.Comparison
		b.WriteString("## Quality Replay: Reference fp16 vs --kv8\n\n")
		fmt.Fprintf(&b, "- Reference (`%s`): **%d/%d (%.1f%%)** passed, avg wall **%.1f ms**, KV width **%d B/elem**, per-lane reserve **%.2f GiB**, max lanes **%d**\n",
			c.ReferenceName, c.RefPassed, c.CardsTotal, c.RefPassRate*100, c.RefAvgWallMs, c.RefKVBytesPerElem, c.RefPerLaneWorstCaseGiB, c.RefMaxAdmittedLanes)
		fmt.Fprintf(&b, "- Candidate (`%s`): **%d/%d (%.1f%%)** passed, avg wall **%.1f ms**, KV width **%d B/elem** (**%.1fx reduction**), per-lane reserve **%.2f GiB** (**%.2fx saving**), max lanes **%d**\n",
			c.CandidateName, c.CandPassed, c.CardsTotal, c.CandPassRate*100, c.CandAvgWallMs, c.CandKVBytesPerElem, c.KVElementReductionRatio, c.CandPerLaneWorstCaseGiB, c.PerLaneMemorySavingRatio, c.CandMaxAdmittedLanes)
		fmt.Fprintf(&b, "- Agreement at `temp=0`: check parity **%.1f%%**, exact output match **%.1f%%**, mean lexical token agreement **%.1f%%**, mean prefix match **%.1f%%**\n\n",
			c.CheckParityRate*100, c.ExactOutputAgreement*100, c.MeanTokenAgreement*100, c.MeanPrefixMatch*100)
		b.WriteString("| Card | Kind | fp16 | kv8 | Exact | Token Agree | fp16 ms | kv8 ms |\n")
		b.WriteString("| --- | --- | --- | --- | --- | --- | ---: | ---: |\n")
		for _, cr := range c.Cards {
			fmt.Fprintf(&b, "| `%s` | `%s` | %v | %v | %v | %.1f%% | %.1f | %.1f |\n",
				cr.CardID, cr.Kind, passLabel(cr.RefPassed), passLabel(cr.CandPassed), cr.ExactMatch, cr.TokenAgreementRatio*100, cr.RefWallMs, cr.CandWallMs)
		}
		b.WriteByte('\n')
	}
	if report.EffortSweep != nil {
		e := report.EffortSweep
		b.WriteString("## Thinking Effort Sweep\n\n")
		b.WriteString("| Effort | Kind | Passed | Pass Rate | Avg Gen Tokens | Avg Wall (ms) |\n")
		b.WriteString("| --- | --- | --- | ---: | ---: | ---: |\n")
		for _, cell := range e.Cells {
			fmt.Fprintf(&b, "| `%s` | `%s` | %d/%d | %.1f%% | %.1f | %.1f |\n",
				cell.ReasoningEffort, cell.Kind, cell.CardsPassed, cell.CardsTotal, cell.PassRate*100, cell.AvgCompTokens, cell.AvgWallMs)
		}
		b.WriteString("\n### Lowest 100%-Passing Effort per Card Kind\n\n")
		for _, rec := range e.Recommendations {
			eff := rec.LowestPassingEffort
			if eff == "" {
				eff = "none-passed-100%"
			}
			fmt.Fprintf(&b, "- `%s`: **%s** (avg %.1f tokens, %.1f ms)\n",
				rec.Kind, eff, rec.AvgTokensAtLowest, rec.AvgWallMsAtLowest)
		}
	}
	return b.String()
}

func passLabel(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}
