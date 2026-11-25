package main

import "encoding/json"

// CardKind classifies benchmark tasks by reasoning complexity.
const (
	KindMechanicalEdit     = "mechanical-edit"
	KindBugfix             = "bugfix"
	KindMultiStepAlgorithm = "multi-step-algorithm"
)

// Message mirrors an OpenAI-compatible chat message.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

// ToolFunction represents a function call name and JSON arguments string.
type ToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCall represents an assistant tool invocation returned by inky serve.
type ToolCall struct {
	Index    int          `json:"index,omitempty"`
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// CheckSpec defines deterministic verification assertions for a benchmark card.
type CheckSpec struct {
	RequireValidToolCall bool              `json:"require_valid_tool_call,omitempty"`
	ExpectedToolName     string            `json:"expected_tool_name,omitempty"`
	RequiredToolArgs     []string          `json:"required_tool_args,omitempty"`
	ToolArgEquals        map[string]string `json:"tool_arg_equals,omitempty"`
	ToolArgDiffField     string            `json:"tool_arg_diff_field,omitempty"`
	ToolArgCodeField     string            `json:"tool_arg_code_field,omitempty"`
	RequireUnifiedDiff   bool              `json:"require_unified_diff,omitempty"`
	ApplyDiffToSource    bool              `json:"apply_diff_to_source,omitempty"`
	RequireValidGoParse  bool              `json:"require_valid_go_parse,omitempty"`
	ExpectedGoSymbols    []string          `json:"expected_go_symbols,omitempty"`
	MustContain          []string          `json:"must_contain,omitempty"`
	MustNotContain       []string          `json:"must_not_contain,omitempty"`
	MustMatchRegex       []string          `json:"must_match_regex,omitempty"`
}

// Card defines one curated quality benchmark card.
type Card struct {
	ID            string            `json:"id"`
	Kind          string            `json:"kind"`
	Title         string            `json:"title"`
	Messages      []Message         `json:"messages"`
	Tools         []json.RawMessage `json:"tools,omitempty"`
	SourceFile    string            `json:"source_file,omitempty"`
	SourceContent string            `json:"source_content,omitempty"`
	MaxTokens     int               `json:"max_tokens,omitempty"`
	Checks        CheckSpec         `json:"checks"`
}

// HealthInfo captures KV cache and memory telemetry from GET /health.
type HealthInfo struct {
	OK                         bool   `json:"ok"`
	ActiveLanes                int    `json:"active_lanes"`
	Inflight                   int    `json:"inflight"`
	RecommendedWorkingSetBytes uint64 `json:"recommended_working_set_bytes"`
	ModelBytes                 uint64 `json:"model_bytes"`
	PerLaneWorstCaseBytes      uint64 `json:"per_lane_worst_case_bytes"`
	MaxAdmittedLanes           int    `json:"max_admitted_lanes"`
	KV8                        bool   `json:"kv8"`
	KVBytesPerElem             int    `json:"kv_bytes_per_elem"`
	ProcessFootprintBytes      uint64 `json:"process_footprint_bytes"`
}

// ChatResult captures one server response and its timing/usage metadata.
type ChatResult struct {
	Content          string     `json:"content"`
	Reasoning        string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	FinishReason     string     `json:"finish_reason"`
	PromptTokens     int        `json:"prompt_tokens"`
	CompletionTokens int        `json:"completion_tokens"`
	CachedTokens     int        `json:"cached_tokens"`
	WallMs           float64    `json:"wall_ms"`
	QueueMs          int64      `json:"queue_ms"`
	PrefillMs        int64      `json:"prefill_ms"`
	DecodeMs         int64      `json:"decode_ms"`
}

// CheckOutcome records the pass/fail result of a single named verification check.
type CheckOutcome struct {
	Name   string `json:"name"`
	Passed bool   `json:"passed"`
	Detail string `json:"detail,omitempty"`
}

// CardResult captures the execution metrics and verification outcome for one card.
type CardResult struct {
	CardID           string         `json:"card_id"`
	Kind             string         `json:"kind"`
	Passed           bool           `json:"passed"`
	ChecksPassed     int            `json:"checks_passed"`
	ChecksTotal      int            `json:"checks_total"`
	Outcomes         []CheckOutcome `json:"outcomes"`
	Error            string         `json:"error,omitempty"`
	NormalizedOutput string         `json:"normalized_output"`
	Response         ChatResult     `json:"response"`
}

// ModeRun captures all card results and server health for one (kv_mode, effort) evaluation.
type ModeRun struct {
	Name            string       `json:"name"`
	KVCacheMode     string       `json:"kv_cache_mode"`
	ReasoningEffort string       `json:"reasoning_effort"`
	Temperature     float64      `json:"temperature"`
	Seed            int64        `json:"seed"`
	Health          HealthInfo   `json:"health"`
	CardsTotal      int          `json:"cards_total"`
	CardsPassed     int          `json:"cards_passed"`
	PassRate        float64      `json:"pass_rate"`
	AvgWallMs       float64      `json:"avg_wall_ms"`
	TotalWallMs     float64      `json:"total_wall_ms"`
	AvgDecodeMs     float64      `json:"avg_decode_ms"`
	TotalCompTokens int          `json:"total_completion_tokens"`
	AvgCompTokens   float64      `json:"avg_completion_tokens"`
	Results         []CardResult `json:"results"`
}

// CardAgreement summarizes fp16 vs kv8 agreement for a single card at temp=0.
type CardAgreement struct {
	CardID              string  `json:"card_id"`
	Kind                string  `json:"kind"`
	RefPassed           bool    `json:"ref_passed"`
	CandPassed          bool    `json:"cand_passed"`
	ExactMatch          bool    `json:"exact_match"`
	TokenAgreementRatio float64 `json:"token_agreement_ratio"`
	PrefixMatchRatio    float64 `json:"prefix_match_ratio"`
	RefCompTokens       int     `json:"ref_completion_tokens"`
	CandCompTokens      int     `json:"cand_completion_tokens"`
	RefWallMs           float64 `json:"ref_wall_ms"`
	CandWallMs          float64 `json:"cand_wall_ms"`
}

// ModeComparison summarizes reference fp16 vs candidate kv8 quality, agreement, memory, and latency.
type ModeComparison struct {
	ReferenceName            string          `json:"reference_name"`
	CandidateName            string          `json:"candidate_name"`
	CardsTotal               int             `json:"cards_total"`
	RefPassed                int             `json:"ref_passed"`
	CandPassed               int             `json:"cand_passed"`
	RefPassRate              float64         `json:"ref_pass_rate"`
	CandPassRate             float64         `json:"cand_pass_rate"`
	CheckParityRate          float64         `json:"check_parity_rate"`
	ExactOutputAgreement     float64         `json:"exact_output_agreement"`
	MeanTokenAgreement       float64         `json:"mean_token_agreement"`
	MeanPrefixMatch          float64         `json:"mean_prefix_match"`
	RefKVBytesPerElem        int             `json:"ref_kv_bytes_per_elem"`
	CandKVBytesPerElem       int             `json:"cand_kv_bytes_per_elem"`
	KVElementReductionRatio  float64         `json:"kv_element_reduction_ratio"`
	RefPerLaneWorstCaseGiB   float64         `json:"ref_per_lane_worst_case_gib"`
	CandPerLaneWorstCaseGiB  float64         `json:"cand_per_lane_worst_case_gib"`
	PerLaneMemorySavingRatio float64         `json:"per_lane_memory_saving_ratio"`
	RefMaxAdmittedLanes      int             `json:"ref_max_admitted_lanes"`
	CandMaxAdmittedLanes     int             `json:"cand_max_admitted_lanes"`
	RefAvgWallMs             float64         `json:"ref_avg_wall_ms"`
	CandAvgWallMs            float64         `json:"cand_avg_wall_ms"`
	RefAvgDecodeMs           float64         `json:"ref_avg_decode_ms"`
	CandAvgDecodeMs          float64         `json:"cand_avg_decode_ms"`
	Cards                    []CardAgreement `json:"cards"`
}

// EffortKindMetrics captures pass rate, token count, and latency for one (effort, kind) cell.
type EffortKindMetrics struct {
	ReasoningEffort     string  `json:"reasoning_effort"`
	Kind                string  `json:"kind"`
	CardsTotal          int     `json:"cards_total"`
	CardsPassed         int     `json:"cards_passed"`
	PassRate            float64 `json:"pass_rate"`
	TotalCompTokens     int     `json:"total_completion_tokens"`
	AvgCompTokens       float64 `json:"avg_completion_tokens"`
	TotalWallMs         float64 `json:"total_wall_ms"`
	AvgWallMs           float64 `json:"avg_wall_ms"`
}

// KindEffortRecommendation identifies the lowest thinking effort passing 100% of checks for a card kind.
type KindEffortRecommendation struct {
	Kind                string  `json:"kind"`
	LowestPassingEffort string  `json:"lowest_passing_effort"`
	AvgTokensAtLowest   float64 `json:"avg_tokens_at_lowest"`
	AvgWallMsAtLowest   float64 `json:"avg_wall_ms_at_lowest"`
}

// EffortSweepReport aggregates effort-sweep runs across card kinds.
type EffortSweepReport struct {
	Efforts         []string                   `json:"efforts"`
	Cells           []EffortKindMetrics        `json:"cells"`
	Recommendations []KindEffortRecommendation `json:"recommendations"`
}

// FullReport combines reference vs kv8 comparison and the thinking-effort sweep.
type FullReport struct {
	Hardware       string             `json:"hardware,omitempty"`
	Model          string             `json:"model,omitempty"`
	Comparison     *ModeComparison    `json:"kv_comparison,omitempty"`
	EffortSweep    *EffortSweepReport `json:"effort_sweep,omitempty"`
	Runs           []ModeRun          `json:"runs"`
}
