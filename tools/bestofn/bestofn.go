package bestofn

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
	"regexp"
	"strings"
	"sync"
	"time"

	"os/exec"
)

// Status values for an individual attempt.
const (
	StatusPassed       = "passed"
	StatusFailedCheck  = "failed_check"
	StatusRequestError = "request_error"
	StatusCancelled    = "cancelled"
)

// Config controls how best-of-N attempts are dispatched and checked.
type Config struct {
	Endpoint          string
	APIKey            string
	Model             string
	N                 int
	Temperature       float64
	MaxTokens         int
	BaseSeed          uint64
	Timeout           time.Duration
	CheckTimeout      time.Duration
	SystemPrompt      string
	CheckCmd          string
	WorkDir           string
	Stream            bool
	CancelOnFirstPass bool
	HTTPClient        *http.Client
}

// Card describes one coding task and its deterministic verification command.
type Card struct {
	ID           string            `json:"id"`
	Tier         string            `json:"tier,omitempty"`
	SystemPrompt string            `json:"system_prompt,omitempty"`
	Prompt       string            `json:"prompt"`
	Files        map[string]string `json:"files,omitempty"`
	TargetFile   string            `json:"target_file,omitempty"`
	CheckCmd     string            `json:"check_cmd,omitempty"`
}

// AttemptResult records the outcome of one parallel lane.
type AttemptResult struct {
	Index            int           `json:"index"`
	Seed             uint64        `json:"seed"`
	Status           string        `json:"status"`
	Content          string        `json:"content,omitempty"`
	PromptTokens     int           `json:"prompt_tokens"`
	CachedTokens     int           `json:"cached_tokens"`
	CompletionTokens int           `json:"completion_tokens"`
	CheckExitCode    int           `json:"check_exit_code"`
	CheckOutput      string        `json:"check_output,omitempty"`
	GenDuration      time.Duration `json:"gen_duration_ns"`
	CheckDuration    time.Duration `json:"check_duration_ns"`
	TotalDuration    time.Duration `json:"total_duration_ns"`
	Error            string        `json:"error,omitempty"`
}

// CardResult summarizes a best-of-N run on a single card.
type CardResult struct {
	CardID            string          `json:"card_id"`
	Tier              string          `json:"tier,omitempty"`
	N                 int             `json:"n"`
	WinnerIndex       int             `json:"winner_index"`
	PassAt1           bool            `json:"pass_at_1"`
	BestOfNPassed     bool            `json:"best_of_n_passed"`
	CancelledLanes    int             `json:"cancelled_lanes"`
	TotalPromptTokens int             `json:"total_prompt_tokens"`
	TotalCachedTokens int             `json:"total_cached_tokens"`
	PrefixCacheRate   float64         `json:"prefix_cache_reuse_rate"`
	WallDuration      time.Duration   `json:"wall_duration_ns"`
	Attempts          []AttemptResult `json:"attempts"`
}

// TierSummary aggregates pass@1 vs best-of-N land rates for one tier.
type TierSummary struct {
	Tier         string  `json:"tier"`
	Cards        int     `json:"cards"`
	PassAt1Count int     `json:"pass_at_1_count"`
	PassAt1Rate  float64 `json:"pass_at_1_rate"`
	BestOfNCount int     `json:"best_of_n_count"`
	BestOfNRate  float64 `json:"best_of_n_rate"`
}

// Report aggregates results across one or more cards.
type Report struct {
	N                   int           `json:"n"`
	Temperature         float64       `json:"temperature"`
	Cards               int           `json:"cards"`
	PassAt1Count        int           `json:"pass_at_1_count"`
	PassAt1Rate         float64       `json:"pass_at_1_rate"`
	BestOfNCount        int           `json:"best_of_n_count"`
	BestOfNRate         float64       `json:"best_of_n_rate"`
	TotalPromptTokens   int           `json:"total_prompt_tokens"`
	TotalCachedTokens   int           `json:"total_cached_tokens"`
	PrefixCacheRate     float64       `json:"prefix_cache_reuse_rate"`
	TotalCancelledLanes int           `json:"total_cancelled_lanes"`
	TotalWallDuration   time.Duration `json:"total_wall_duration_ns"`
	ByTier              []TierSummary `json:"by_tier,omitempty"`
	Results             []CardResult  `json:"results"`
}

// DefaultConfig returns sensible defaults for parallel best-of-N coding runs.
func DefaultConfig() Config {
	return Config{
		Endpoint:          "http://127.0.0.1:8090",
		Model:             "inkling-2-mini",
		N:                 4,
		Temperature:       0.6,
		MaxTokens:         1024,
		BaseSeed:          1001,
		Timeout:           180 * time.Second,
		CheckTimeout:      30 * time.Second,
		Stream:            true,
		CancelOnFirstPass: true,
	}
}

func (c Config) validate() error {
	if strings.TrimSpace(c.Endpoint) == "" {
		return errors.New("endpoint must not be empty")
	}
	if c.N <= 0 {
		return fmt.Errorf("n must be positive, got %d", c.N)
	}
	if c.Temperature < 0 || c.Temperature > 2 {
		return fmt.Errorf("temperature must be in [0, 2], got %v", c.Temperature)
	}
	if c.MaxTokens <= 0 {
		return fmt.Errorf("max tokens must be positive, got %d", c.MaxTokens)
	}
	return nil
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatRequest struct {
	Model         string         `json:"model"`
	Messages      []chatMessage  `json:"messages"`
	Temperature   float64        `json:"temperature"`
	MaxTokens     int            `json:"max_tokens"`
	Seed          uint64         `json:"seed"`
	Stream        bool           `json:"stream"`
	StreamOptions *streamOptions `json:"stream_options,omitempty"`
}

type promptTokenDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type usagePayload struct {
	PromptTokens        int                 `json:"prompt_tokens"`
	CompletionTokens    int                 `json:"completion_tokens"`
	TotalTokens         int                 `json:"total_tokens"`
	PromptTokensDetails *promptTokenDetails `json:"prompt_tokens_details,omitempty"`
}

type chatChoice struct {
	Index   int `json:"index"`
	Message *struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message,omitempty"`
	Delta *struct {
		Content string `json:"content"`
	} `json:"delta,omitempty"`
	FinishReason *string `json:"finish_reason,omitempty"`
}

type chatResponse struct {
	Choices []chatChoice  `json:"choices"`
	Usage   *usagePayload `json:"usage,omitempty"`
}

// RunCard dispatches N parallel attempts sharing one prompt prefix and returns as soon as
// the first attempt passes its check (cancelling remaining lanes when CancelOnFirstPass is true).
func RunCard(ctx context.Context, cfg Config, card Card) (CardResult, error) {
	if err := cfg.validate(); err != nil {
		return CardResult{}, err
	}
	checkCmd := strings.TrimSpace(card.CheckCmd)
	if checkCmd == "" {
		checkCmd = strings.TrimSpace(cfg.CheckCmd)
	}
	if checkCmd == "" {
		return CardResult{}, errors.New("check command must not be empty")
	}
	if strings.TrimSpace(card.Prompt) == "" {
		return CardResult{}, errors.New("card prompt must not be empty")
	}

	sysPrompt := card.SystemPrompt
	if sysPrompt == "" {
		sysPrompt = cfg.SystemPrompt
	}
	// Shared verbatim messages slice so every lane hits the exact same prompt prefix.
	messages := make([]chatMessage, 0, 2)
	if sysPrompt != "" {
		messages = append(messages, chatMessage{Role: "system", Content: sysPrompt})
	}
	messages = append(messages, chatMessage{Role: "user", Content: card.Prompt})

	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{}
	}

	cardCtx := ctx
	var cancel context.CancelFunc
	if cfg.Timeout > 0 {
		cardCtx, cancel = context.WithTimeout(ctx, cfg.Timeout)
	} else {
		cardCtx, cancel = context.WithCancel(ctx)
	}
	defer cancel()

	start := time.Now()
	attempts := make([]AttemptResult, cfg.N)
	var winnerMu sync.Mutex
	winnerIdx := -1
	var wg sync.WaitGroup

	for i := 0; i < cfg.N; i++ {
		wg.Add(1)
		go func(lane int) {
			defer wg.Done()
			seed := cfg.BaseSeed + uint64(lane)
			res := runAttempt(cardCtx, client, cfg, card, messages, checkCmd, lane, seed)
			attempts[lane] = res
			if res.Status == StatusPassed {
				winnerMu.Lock()
				if winnerIdx < 0 {
					winnerIdx = lane
					if cfg.CancelOnFirstPass {
						cancel()
					}
				}
				winnerMu.Unlock()
			}
		}(i)
	}
	wg.Wait()

	wall := time.Since(start)
	out := CardResult{
		CardID:        card.ID,
		Tier:          card.Tier,
		N:             cfg.N,
		WinnerIndex:   winnerIdx,
		PassAt1:       len(attempts) > 0 && attempts[0].Status == StatusPassed,
		BestOfNPassed: winnerIdx >= 0,
		WallDuration:  wall,
		Attempts:      attempts,
	}
	for _, a := range attempts {
		out.TotalPromptTokens += a.PromptTokens
		out.TotalCachedTokens += a.CachedTokens
		if a.Status == StatusCancelled {
			out.CancelledLanes++
		}
	}
	if out.TotalPromptTokens > 0 {
		out.PrefixCacheRate = float64(out.TotalCachedTokens) / float64(out.TotalPromptTokens)
	}
	return out, nil
}

// RunSuite executes a slice of cards sequentially (each with N parallel lanes) and summarizes
// overall and per-tier pass@1 vs best-of-N land rates.
func RunSuite(ctx context.Context, cfg Config, cards []Card) (Report, error) {
	if err := cfg.validate(); err != nil {
		return Report{}, err
	}
	rep := Report{
		N:           cfg.N,
		Temperature: cfg.Temperature,
		Cards:       len(cards),
		Results:     make([]CardResult, 0, len(cards)),
	}
	tierOrder := make([]string, 0)
	tierMap := make(map[string]*TierSummary)
	suiteStart := time.Now()

	for i, card := range cards {
		if card.ID == "" {
			card.ID = fmt.Sprintf("card-%d", i+1)
		}
		res, err := RunCard(ctx, cfg, card)
		if err != nil {
			return Report{}, fmt.Errorf("card %s: %w", card.ID, err)
		}
		rep.Results = append(rep.Results, res)
		if res.PassAt1 {
			rep.PassAt1Count++
		}
		if res.BestOfNPassed {
			rep.BestOfNCount++
		}
		rep.TotalPromptTokens += res.TotalPromptTokens
		rep.TotalCachedTokens += res.TotalCachedTokens
		rep.TotalCancelledLanes += res.CancelledLanes

		tierName := res.Tier
		if tierName == "" {
			tierName = "default"
		}
		ts, ok := tierMap[tierName]
		if !ok {
			ts = &TierSummary{Tier: tierName}
			tierMap[tierName] = ts
			tierOrder = append(tierOrder, tierName)
		}
		ts.Cards++
		if res.PassAt1 {
			ts.PassAt1Count++
		}
		if res.BestOfNPassed {
			ts.BestOfNCount++
		}
	}

	rep.TotalWallDuration = time.Since(suiteStart)
	if rep.Cards > 0 {
		rep.PassAt1Rate = float64(rep.PassAt1Count) / float64(rep.Cards)
		rep.BestOfNRate = float64(rep.BestOfNCount) / float64(rep.Cards)
	}
	if rep.TotalPromptTokens > 0 {
		rep.PrefixCacheRate = float64(rep.TotalCachedTokens) / float64(rep.TotalPromptTokens)
	}
	for _, name := range tierOrder {
		ts := tierMap[name]
		if ts.Cards > 0 {
			ts.PassAt1Rate = float64(ts.PassAt1Count) / float64(ts.Cards)
			ts.BestOfNRate = float64(ts.BestOfNCount) / float64(ts.Cards)
		}
		rep.ByTier = append(rep.ByTier, *ts)
	}
	return rep, nil
}

func runAttempt(
	ctx context.Context,
	client *http.Client,
	cfg Config,
	card Card,
	messages []chatMessage,
	checkCmd string,
	lane int,
	seed uint64,
) AttemptResult {
	t0 := time.Now()
	res := AttemptResult{
		Index:         lane,
		Seed:          seed,
		CheckExitCode: -1,
	}

	content, usage, err := callChatCompletion(ctx, client, cfg, messages, seed)
	res.GenDuration = time.Since(t0)
	if usage != nil {
		res.PromptTokens = usage.PromptTokens
		res.CompletionTokens = usage.CompletionTokens
		if usage.PromptTokensDetails != nil {
			res.CachedTokens = usage.PromptTokensDetails.CachedTokens
		}
	}
	if err != nil {
		if ctx.Err() != nil || errors.Is(err, context.Canceled) {
			res.Status = StatusCancelled
		} else {
			res.Status = StatusRequestError
			res.Error = err.Error()
		}
		res.TotalDuration = time.Since(t0)
		return res
	}
	res.Content = content

	if ctx.Err() != nil {
		res.Status = StatusCancelled
		res.TotalDuration = time.Since(t0)
		return res
	}

	checkStart := time.Now()
	exitCode, output, checkErr := runDeterministicCheck(ctx, cfg, card, content, lane, seed)
	res.CheckDuration = time.Since(checkStart)
	res.CheckExitCode = exitCode
	res.CheckOutput = output
	res.TotalDuration = time.Since(t0)

	if checkErr != nil {
		if ctx.Err() != nil || errors.Is(checkErr, context.Canceled) {
			res.Status = StatusCancelled
			return res
		}
		res.Status = StatusFailedCheck
		res.Error = checkErr.Error()
		return res
	}
	if exitCode == 0 {
		res.Status = StatusPassed
	} else {
		res.Status = StatusFailedCheck
	}
	return res
}

func callChatCompletion(
	ctx context.Context,
	client *http.Client,
	cfg Config,
	messages []chatMessage,
	seed uint64,
) (string, *usagePayload, error) {
	reqBody := chatRequest{
		Model:       cfg.Model,
		Messages:    messages,
		Temperature: cfg.Temperature,
		MaxTokens:   cfg.MaxTokens,
		Seed:        seed,
		Stream:      cfg.Stream,
	}
	if cfg.Stream {
		reqBody.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return "", nil, fmt.Errorf("marshal chat request: %w", err)
	}

	url := strings.TrimRight(cfg.Endpoint, "/") + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return "", nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	if !cfg.Stream {
		var parsed chatResponse
		if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
			return "", nil, fmt.Errorf("decode response: %w", err)
		}
		if len(parsed.Choices) == 0 || parsed.Choices[0].Message == nil {
			return "", parsed.Usage, errors.New("response contained no choices")
		}
		return parsed.Choices[0].Message.Content, parsed.Usage, nil
	}

	var content strings.Builder
	var usage *usagePayload
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk chatResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return content.String(), usage, fmt.Errorf("decode SSE chunk: %w", err)
		}
		if chunk.Usage != nil {
			u := *chunk.Usage
			usage = &u
		}
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta != nil {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}
	if err := scanner.Err(); err != nil {
		return content.String(), usage, err
	}
	return content.String(), usage, nil
}

func runDeterministicCheck(
	ctx context.Context,
	cfg Config,
	card Card,
	content string,
	lane int,
	seed uint64,
) (int, string, error) {
	tmpDir, err := os.MkdirTemp("", fmt.Sprintf("bestofn-lane%d-*", lane))
	if err != nil {
		return -1, "", fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	if cfg.WorkDir != "" {
		if err := copyTree(cfg.WorkDir, tmpDir); err != nil {
			return -1, "", fmt.Errorf("copy workdir: %w", err)
		}
	}
	for rel, body := range card.Files {
		if err := writeIsolatedFile(tmpDir, rel, body); err != nil {
			return -1, "", fmt.Errorf("materialize card file %q: %w", rel, err)
		}
	}

	respPath := filepath.Join(tmpDir, ".bestofn_response.txt")
	if err := os.WriteFile(respPath, []byte(content), 0o600); err != nil {
		return -1, "", fmt.Errorf("write response file: %w", err)
	}

	extracted := ExtractFiles(content, card.TargetFile)
	for rel, body := range extracted {
		if err := writeIsolatedFile(tmpDir, rel, body); err != nil {
			return -1, "", fmt.Errorf("write extracted file %q: %w", rel, err)
		}
	}

	checkCtx := ctx
	var cancel context.CancelFunc
	if cfg.CheckTimeout > 0 {
		checkCtx, cancel = context.WithTimeout(ctx, cfg.CheckTimeout)
		defer cancel()
	}

	cmd := exec.CommandContext(checkCtx, "sh", "-c", cardCheckCmd(cfg, card))
	cmd.Dir = tmpDir
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("BESTOFN_ATTEMPT=%d", lane),
		fmt.Sprintf("BESTOFN_SEED=%d", seed),
		"BESTOFN_RESPONSE_FILE="+respPath,
	)
	var outBuf limitedBuffer
	outBuf.limit = 64 * 1024
	cmd.Stdout = &outBuf
	cmd.Stderr = &outBuf

	err = cmd.Run()
	if err == nil {
		return 0, outBuf.String(), nil
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode(), outBuf.String(), nil
	}
	return -1, outBuf.String(), err
}

func cardCheckCmd(cfg Config, card Card) string {
	if s := strings.TrimSpace(card.CheckCmd); s != "" {
		return s
	}
	return strings.TrimSpace(cfg.CheckCmd)
}

var fencedBlockRe = regexp.MustCompile("(?s)```([^\\n`]*)\\n(.*?)```")

// ExtractFiles extracts either explicit `lang:path` / `file:path` fenced blocks or, if targetFile
// is set, the first fenced code block (falling back to the trimmed raw response).
func ExtractFiles(content, targetFile string) map[string]string {
	files := make(map[string]string)
	matches := fencedBlockRe.FindAllStringSubmatch(content, -1)
	var firstCode string
	for _, m := range matches {
		header := strings.TrimSpace(m[1])
		body := m[2]
		if firstCode == "" {
			firstCode = body
		}
		if path := parseBlockPath(header); path != "" {
			files[path] = body
		}
	}
	if len(files) > 0 {
		return files
	}
	if targetFile != "" {
		if firstCode != "" {
			files[targetFile] = firstCode
		} else {
			trimmed := strings.TrimSpace(content)
			if strings.HasPrefix(trimmed, "```") {
				if nl := strings.IndexByte(trimmed, '\n'); nl >= 0 {
					trimmed = strings.TrimSuffix(strings.TrimSpace(trimmed[nl+1:]), "```")
				}
			}
			files[targetFile] = strings.TrimSpace(trimmed) + "\n"
		}
	}
	return files
}

func parseBlockPath(header string) string {
	if header == "" {
		return ""
	}
	for _, tok := range strings.Fields(header) {
		if strings.HasPrefix(tok, "file:") {
			return strings.TrimPrefix(tok, "file:")
		}
		if idx := strings.IndexByte(tok, ':'); idx > 0 && idx+1 < len(tok) {
			return tok[idx+1:]
		}
	}
	return ""
}

func writeIsolatedFile(root, rel, body string) error {
	clean := filepath.Clean(rel)
	if filepath.IsAbs(clean) || clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return fmt.Errorf("unsafe relative path %q", rel)
	}
	dst := filepath.Join(root, clean)
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	return os.WriteFile(dst, []byte(body), 0o644)
}

func copyTree(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(target, info.Mode().Perm())
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, info.Mode().Perm())
	})
}

type limitedBuffer struct {
	buf   bytes.Buffer
	limit int
}

func (l *limitedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	if l.limit > 0 && l.buf.Len() < l.limit {
		rem := l.limit - l.buf.Len()
		if len(p) > rem {
			p = p[:rem]
		}
		l.buf.Write(p)
	}
	return n, nil
}

func (l *limitedBuffer) String() string { return l.buf.String() }

// FormatSummary renders a concise human-readable report of pass@1 vs best-of-N land rates,
// prefix-cache reuse, and wall-clock time.
func FormatSummary(rep Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "bestofn: n=%d temp=%.2f cards=%d pass@1=%d/%d (%.1f%%) best-of-%d=%d/%d (%.1f%%)\n",
		rep.N, rep.Temperature, rep.Cards,
		rep.PassAt1Count, rep.Cards, rep.PassAt1Rate*100,
		rep.N, rep.BestOfNCount, rep.Cards, rep.BestOfNRate*100)
	fmt.Fprintf(&b, "prefix-cache: cached=%d prompt=%d reuse=%.1f%% cancelled_lanes=%d wall=%s\n",
		rep.TotalCachedTokens, rep.TotalPromptTokens, rep.PrefixCacheRate*100,
		rep.TotalCancelledLanes, rep.TotalWallDuration.Round(time.Millisecond))
	for _, ts := range rep.ByTier {
		fmt.Fprintf(&b, "  tier %-10s cards=%d pass@1=%d/%d (%.1f%%) best-of-%d=%d/%d (%.1f%%)\n",
			ts.Tier, ts.Cards, ts.PassAt1Count, ts.Cards, ts.PassAt1Rate*100,
			rep.N, ts.BestOfNCount, ts.Cards, ts.BestOfNRate*100)
	}
	return b.String()
}
