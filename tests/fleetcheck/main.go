// fleetcheck exercises live multi-turn coding-agent cards with real tool calls
// (read_file -> edit_file -> run_test) against `inky serve` over a private network,
// and drives multi-turn card recording traffic for `tests/replay/recorded_cards.jsonl`.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type toolDef struct {
	Type     string      `json:"type"`
	Function toolFnDescr `json:"function"`
}

type toolFnDescr struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type toolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type chatMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

type chatRequest struct {
	Model           string        `json:"model"`
	Messages        []chatMessage `json:"messages"`
	Tools           []toolDef     `json:"tools,omitempty"`
	MaxTokens       int           `json:"max_tokens"`
	Temperature     float64       `json:"temperature"`
	Seed            uint64        `json:"seed,omitempty"`
	ReasoningEffort string        `json:"reasoning_effort,omitempty"`
	Stream          bool          `json:"stream"`
}

type chatResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Message struct {
			Role      string     `json:"role"`
			Content   string     `json:"content"`
			Reasoning string     `json:"reasoning_content"`
			ToolCalls []toolCall `json:"tool_calls"`
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

func codingTools() []toolDef {
	return []toolDef{
		{
			Type: "function",
			Function: toolFnDescr{
				Name:        "read_file",
				Description: "Read a file inside the workspace by relative path.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{"type": "string", "description": "Relative path inside workspace"},
					},
					"required": []string{"path"},
				},
			},
		},
		{
			Type: "function",
			Function: toolFnDescr{
				Name:        "edit_file",
				Description: "Replace an exact substring old_text with new_text in a workspace file.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path":     map[string]any{"type": "string", "description": "Relative path inside workspace"},
						"old_text": map[string]any{"type": "string", "description": "Exact existing text to replace"},
						"new_text": map[string]any{"type": "string", "description": "Replacement text"},
					},
					"required": []string{"path", "old_text", "new_text"},
				},
			},
		},
		{
			Type: "function",
			Function: toolFnDescr{
				Name:        "run_test",
				Description: "Run tests in the workspace (e.g. 'go test -v ./...').",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"command": map[string]any{"type": "string", "description": "Test command to run"},
					},
					"required": []string{"command"},
				},
			},
		},
	}
}

func safeJoin(root, rel string) (string, error) {
	if rel == "" || filepath.IsAbs(rel) {
		return "", fmt.Errorf("invalid relative path %q", rel)
	}
	clean := filepath.Clean(rel)
	if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes workspace: %q", rel)
	}
	joined := filepath.Join(root, clean)
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	absJoined, err := filepath.Abs(joined)
	if err != nil {
		return "", err
	}
	if absJoined != absRoot && !strings.HasPrefix(absJoined, absRoot+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes workspace: %q", rel)
	}
	return absJoined, nil
}

func execTool(workdir, name, argsJSON string) (string, bool) {
	var args map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return fmt.Sprintf("invalid JSON arguments: %v", err), false
	}
	switch name {
	case "read_file":
		p, _ := args["path"].(string)
		full, err := safeJoin(workdir, p)
		if err != nil {
			return err.Error(), false
		}
		b, err := os.ReadFile(full)
		if err != nil {
			return fmt.Sprintf("read_file error: %v", err), false
		}
		return string(b), true
	case "edit_file":
		p, _ := args["path"].(string)
		oldText, _ := args["old_text"].(string)
		newText, _ := args["new_text"].(string)
		full, err := safeJoin(workdir, p)
		if err != nil {
			return err.Error(), false
		}
		b, err := os.ReadFile(full)
		if err != nil {
			return fmt.Sprintf("edit_file read error: %v", err), false
		}
		src := string(b)
		if !strings.Contains(src, oldText) {
			return fmt.Sprintf("old_text not found in %s", p), false
		}
		updated := strings.Replace(src, oldText, newText, 1)
		if err := os.WriteFile(full, []byte(updated), 0o644); err != nil {
			return fmt.Sprintf("edit_file write error: %v", err), false
		}
		return fmt.Sprintf("edited %s successfully", p), true
	case "run_test":
		cmdStr, _ := args["command"].(string)
		if strings.TrimSpace(cmdStr) == "" {
			cmdStr = "go test -v ./..."
		}
		if cmdStr != "go test ./..." && cmdStr != "go test -v ./..." {
			return fmt.Sprintf("unsupported test command %q (allowed: go test -v ./...)", cmdStr), false
		}
		cmd := exec.Command("go", "test", "-v", "./...")
		cmd.Dir = workdir
		out, err := cmd.CombinedOutput()
		if err != nil {
			return fmt.Sprintf("exit error: %v\n%s", err, string(out)), false
		}
		return string(out), true
	default:
		return fmt.Sprintf("unknown tool %q", name), false
	}
}

func setupFleetWorkspace(dir string) error {
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module fleetcard\n\ngo 1.24\n"), 0o644); err != nil {
		return err
	}
	src := `package fleetcard

// Clamp restricts v to the inclusive range [lo, hi].
func Clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return lo
	}
	return v
}
`
	testSrc := `package fleetcard

import "testing"

func TestClamp(t *testing.T) {
	if got := Clamp(-5, 0, 10); got != 0 {
		t.Fatalf("Clamp(-5, 0, 10) = %d, want 0", got)
	}
	if got := Clamp(25, 0, 10); got != 10 {
		t.Fatalf("Clamp(25, 0, 10) = %d, want 10", got)
	}
	if got := Clamp(5, 0, 10); got != 5 {
		t.Fatalf("Clamp(5, 0, 10) = %d, want 5", got)
	}
}
`
	if err := os.WriteFile(filepath.Join(dir, "mathutil.go"), []byte(src), 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "mathutil_test.go"), []byte(testSrc), 0o644)
}

func sendChat(client *http.Client, baseURL, apiKey, cardID string, reqBody chatRequest) (chatResponse, error) {
	var out chatResponse
	raw, err := json.Marshal(reqBody)
	if err != nil {
		return out, err
	}
	endpoint := strings.TrimRight(baseURL, "/") + "/v1/chat/completions"
	req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(raw))
	if err != nil {
		return out, err
	}
	req.Header.Set("Content-Type", "application/json")
	if cardID != "" {
		req.Header.Set("X-Inky-Card-Id", cardID)
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return out, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	if !reqBody.Stream {
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			return out, err
		}
		if len(out.Choices) == 0 {
			return out, errors.New("empty choices in response")
		}
		return out, nil
	}

	// Stream SSE and collect full message + tool_calls + usage.
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
	type sseChunk struct {
		ID      string `json:"id"`
		Choices []struct {
			Delta struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				Reasoning string `json:"reasoning_content"`
				ToolCalls []struct {
					Index    int    `json:"index"`
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"delta"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	out.Choices = make([]struct {
		Message struct {
			Role      string     `json:"role"`
			Content   string     `json:"content"`
			Reasoning string     `json:"reasoning_content"`
			ToolCalls []toolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	}, 1)
	var tcMap = make(map[int]*toolCall)
	var maxIdx = -1
	var done bool
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		if data == "[DONE]" {
			done = true
			break
		}
		var ch sseChunk
		if err := json.Unmarshal([]byte(data), &ch); err != nil {
			return out, fmt.Errorf("decode sse: %w", err)
		}
		if ch.ID != "" {
			out.ID = ch.ID
		}
		if ch.Usage != nil {
			out.Usage.PromptTokens = ch.Usage.PromptTokens
			out.Usage.CompletionTokens = ch.Usage.CompletionTokens
			out.Usage.PromptTokensDetails.CachedTokens = ch.Usage.PromptTokensDetails.CachedTokens
		}
		if len(ch.Choices) > 0 {
			c := ch.Choices[0]
			if c.Delta.Role != "" {
				out.Choices[0].Message.Role = c.Delta.Role
			}
			out.Choices[0].Message.Content += c.Delta.Content
			out.Choices[0].Message.Reasoning += c.Delta.Reasoning
			if c.FinishReason != "" {
				out.Choices[0].FinishReason = c.FinishReason
			}
			for _, dt := range c.Delta.ToolCalls {
				tc, exists := tcMap[dt.Index]
				if !exists {
					tc = &toolCall{Type: "function"}
					tcMap[dt.Index] = tc
					if dt.Index > maxIdx {
						maxIdx = dt.Index
					}
				}
				if dt.ID != "" {
					tc.ID = dt.ID
				}
				if dt.Type != "" {
					tc.Type = dt.Type
				}
				if dt.Function.Name != "" {
					tc.Function.Name += dt.Function.Name
				}
				if dt.Function.Arguments != "" {
					tc.Function.Arguments += dt.Function.Arguments
				}
			}
		}
	}
	if err := sc.Err(); err != nil {
		return out, err
	}
	if !done {
		return out, errors.New("stream ended without [DONE]")
	}
	for i := 0; i <= maxIdx; i++ {
		if tc, ok := tcMap[i]; ok {
			out.Choices[0].Message.ToolCalls = append(out.Choices[0].Message.ToolCalls, *tc)
		}
	}
	return out, nil
}

func runFleetCard(client *http.Client, baseURL, apiKey, cardID, workBase string) error {
	var dir string
	var cleanup bool
	if workBase != "" {
		dir = workBase
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	} else {
		var err error
		dir, err = os.MkdirTemp("", "inky-fleet-card-*")
		if err != nil {
			return err
		}
		cleanup = true
	}
	if cleanup {
		defer os.RemoveAll(dir)
	}
	if err := setupFleetWorkspace(dir); err != nil {
		return err
	}
	preCmd := exec.Command("go", "test", "./...")
	preCmd.Dir = dir
	if out, err := preCmd.CombinedOutput(); err == nil {
		return fmt.Errorf("expected initial go test to fail, but it passed:\n%s", out)
	}

	messages := []chatMessage{
		{
			Role:    "system",
			Content: "You are a concise Go coding agent. Always use the provided tools read_file, edit_file, and run_test to inspect the code, fix bugs, and verify tests pass before giving your final answer.",
		},
		{
			Role:    "user",
			Content: "In this Go workspace, `go test ./...` fails in `mathutil.go` / `mathutil_test.go`. Use `read_file` on `mathutil.go` and `mathutil_test.go`, fix the bug with `edit_file`, verify with `run_test`, and report the result.",
		},
	}

	seenTools := map[string]bool{}
	var totalPrompt, totalCached, totalCompletion int
	turns := 0
	for turn := 0; turn < 8; turn++ {
		turns++
		resp, err := sendChat(client, baseURL, apiKey, cardID, chatRequest{
			Model:           "inkling-2-mini",
			Messages:        messages,
			Tools:           codingTools(),
			MaxTokens:       1024,
			Temperature:     0.0,
			Seed:            42,
			ReasoningEffort: "low",
			Stream:          false,
		})
		if err != nil {
			return fmt.Errorf("turn %d chat error: %w", turn, err)
		}
		totalPrompt += resp.Usage.PromptTokens
		totalCached += resp.Usage.PromptTokensDetails.CachedTokens
		totalCompletion += resp.Usage.CompletionTokens

		msg := resp.Choices[0].Message
		asst := chatMessage{
			Role:      "assistant",
			Content:   msg.Content,
			ToolCalls: msg.ToolCalls,
		}
		if len(msg.ToolCalls) > 0 && msg.Content == "" {
			asst.Content = nil
		}
		messages = append(messages, asst)

		if len(msg.ToolCalls) == 0 {
			break
		}
		for _, tc := range msg.ToolCalls {
			seenTools[tc.Function.Name] = true
			out, ok := execTool(dir, tc.Function.Name, tc.Function.Arguments)
			fmt.Printf("turn=%d tool=%s ok=%v args=%s\n", turn, tc.Function.Name, ok, tc.Function.Arguments)
			messages = append(messages, chatMessage{
				Role:       "tool",
				ToolCallID: tc.ID,
				Name:       tc.Function.Name,
				Content:    out,
			})
		}
	}

	postCmd := exec.Command("go", "test", "-v", "./...")
	postCmd.Dir = dir
	postOut, err := postCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("post-card go test failed: %v\n%s", err, string(postOut))
	}
	if !seenTools["read_file"] || !seenTools["edit_file"] || !seenTools["run_test"] {
		return fmt.Errorf("expected read_file, edit_file, and run_test tool calls, saw: %v", seenTools)
	}
	fmt.Printf("FLEET_CARD_PASS card_id=%s turns=%d tools=read_file,edit_file,run_test prompt_tokens=%d cached_tokens=%d completion_tokens=%d\n%s",
		cardID, turns, totalPrompt, totalCached, totalCompletion, string(postOut))
	return nil
}

func runRecordCards(client *http.Client, baseURL, apiKey string) error {
	sys := "You are a concise software engineering assistant working in a Go and Rust codebase. Use tools carefully and provide brief, exact responses."
	type cardScript struct {
		id     string
		prompt string
		file   string
		body   string
		follow string
	}
	scripts := []cardScript{
		{
			id:     "card-clamp-fix",
			prompt: "Fix the upper-bound bug in `mathutil.go`. Read `mathutil.go` first.",
			file:   "mathutil.go",
			body:   "package fleetcard\nfunc Clamp(v, lo, hi int) int {\n\tif v < lo { return lo }\n\tif v > hi { return lo }\n\treturn v\n}\n",
			follow: "Now apply the fix with edit_file and verify.",
		},
		{
			id:     "card-slice-bounds",
			prompt: "Inspect `ring.go` for an off-by-one window index check.",
			file:   "ring.go",
			body:   "package ring\nfunc WindowStart(pos, win int) int {\n\tif pos <= win { return 0 }\n\treturn pos - win\n}\n",
			follow: "Check edge cases for pos == win and pos == 0.",
		},
		{
			id:     "card-retry-after",
			prompt: "Review `retry.go` for Retry-After clamping between 0s and 60s.",
			file:   "retry.go",
			body:   "package retry\nimport \"time\"\nfunc ClampDelay(s int) time.Duration {\n\tif s < 0 { s = 0 }\n\tif s > 60 { s = 60 }\n\treturn time.Duration(s) * time.Second\n}\n",
			follow: "Summarize whether negative and >60 values are clamped safely.",
		},
		{
			id:     "card-cidr-check",
			prompt: "Verify RFC 6598 Tailscale CGNAT 100.64.0.0/10 check in `netutil.rs`.",
			file:   "netutil.rs",
			body:   "pub fn is_cgnat(octets: [u8; 4]) -> bool {\n    octets[0] == 100 && (octets[1] & 0b1100_0000) == 64\n}\n",
			follow: "Confirm boundary values 100.63.255.255, 100.64.0.0, 100.127.255.255, 100.128.0.0.",
		},
		{
			id:     "card-health-json",
			prompt: "Inspect `health.go` for active_lanes and queued_requests JSON decoding.",
			file:   "health.go",
			body:   "package health\ntype Snap struct {\n\tOK bool `json:\"ok\"`\n\tActiveLanes int `json:\"active_lanes\"`\n\tQueued int `json:\"queued_requests\"`\n}\n",
			follow: "Suggest one unit test assertion for drained idle state.",
		},
		{
			id:     "card-prefix-cache",
			prompt: "Inspect `prefix.rs` for SWA ring invalidation threshold.",
			file:   "prefix.rs",
			body:   "pub fn keep_valid(kept: usize, pos: usize, swa_win: usize) -> bool {\n    pos < kept.saturating_sub(swa_win)\n}\n",
			follow: "Explain the wrap invariant in two sentences.",
		},
	}

	tools := codingTools()
	for idx, sc := range scripts {
		msgs := []chatMessage{
			{Role: "system", Content: sys},
			{Role: "user", Content: sc.prompt},
		}
		// Turn 0 (streaming)
		r0, err := sendChat(client, baseURL, apiKey, sc.id, chatRequest{
			Model:           "inkling-2-mini",
			Messages:        msgs,
			Tools:           tools,
			MaxTokens:       160,
			Temperature:     0.2,
			Seed:            uint64(100 + idx),
			ReasoningEffort: "low",
			Stream:          true,
		})
		if err != nil {
			return fmt.Errorf("%s turn 0: %w", sc.id, err)
		}
		m0 := r0.Choices[0].Message
		asst0 := chatMessage{Role: "assistant", Content: m0.Content, ToolCalls: m0.ToolCalls}
		if len(m0.ToolCalls) > 0 && m0.Content == "" {
			asst0.Content = nil
		}
		msgs = append(msgs, asst0)
		if len(m0.ToolCalls) > 0 {
			for _, tc := range m0.ToolCalls {
				msgs = append(msgs, chatMessage{
					Role:       "tool",
					ToolCallID: tc.ID,
					Name:       tc.Function.Name,
					Content:    sc.body,
				})
			}
		} else {
			msgs = append(msgs, chatMessage{
				Role:    "user",
				Content: fmt.Sprintf("Here is `%s`:\n```\n%s```\n%s", sc.file, sc.body, sc.follow),
			})
		}

		// Turn 1 (streaming)
		r1, err := sendChat(client, baseURL, apiKey, sc.id, chatRequest{
			Model:           "inkling-2-mini",
			Messages:        msgs,
			Tools:           tools,
			MaxTokens:       160,
			Temperature:     0.2,
			Seed:            uint64(200 + idx),
			ReasoningEffort: "low",
			Stream:          true,
		})
		if err != nil {
			return fmt.Errorf("%s turn 1: %w", sc.id, err)
		}
		m1 := r1.Choices[0].Message
		asst1 := chatMessage{Role: "assistant", Content: m1.Content, ToolCalls: m1.ToolCalls}
		if len(m1.ToolCalls) > 0 && m1.Content == "" {
			asst1.Content = nil
		}
		msgs = append(msgs, asst1)
		if len(m1.ToolCalls) > 0 {
			for _, tc := range m1.ToolCalls {
				msgs = append(msgs, chatMessage{
					Role:       "tool",
					ToolCallID: tc.ID,
					Name:       tc.Function.Name,
					Content:    "PASS: ok  fleetcard 0.012s",
				})
			}
		} else {
			msgs = append(msgs, chatMessage{
				Role:    "user",
				Content: "Tests passed (`ok 0.012s`). Give a 1-sentence final summary.",
			})
		}

		// Turn 2 (mix stream & non-stream)
		_, err = sendChat(client, baseURL, apiKey, sc.id, chatRequest{
			Model:           "inkling-2-mini",
			Messages:        msgs,
			Tools:           tools,
			MaxTokens:       96,
			Temperature:     0.2,
			Seed:            uint64(300 + idx),
			ReasoningEffort: "low",
			Stream:          idx%2 == 0,
		})
		if err != nil {
			return fmt.Errorf("%s turn 2: %w", sc.id, err)
		}
		fmt.Printf("RECORDED_CARD id=%s turns=3\n", sc.id)
	}
	return nil
}

func main() {
	mode := flag.String("mode", "fleet-card", "fleet-card or record-cards")
	url := flag.String("url", "http://127.0.0.1:18096", "base URL of inky serve")
	keyEnv := flag.String("key-env", "INKY_SERVE_KEY", "environment variable holding bearer key (empty for unauthenticated)")
	cardID := flag.String("card-id", "fleet-card-live-1", "X-Inky-Card-Id header value")
	workdir := flag.String("workdir", "", "optional workspace dir for fleet-card")
	flag.Parse()

	var apiKey string
	if *keyEnv != "" {
		apiKey = os.Getenv(*keyEnv)
	}
	client := &http.Client{Timeout: 3 * time.Minute}

	var err error
	switch *mode {
	case "fleet-card":
		err = runFleetCard(client, *url, apiKey, *cardID, *workdir)
	case "record-cards":
		err = runRecordCards(client, *url, apiKey)
	default:
		err = fmt.Errorf("unknown mode %q", *mode)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "fleetcheck failed: %v\n", err)
		os.Exit(1)
	}
}
