package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"regexp"
	"strconv"
	"strings"
)

// UnifiedHunk represents one @@ -a,b +c,d @@ hunk in a unified diff.
type UnifiedHunk struct {
	OldStart int
	OldCount int
	NewStart int
	NewCount int
	Lines    []string
}

// UnifiedDiff represents a parsed unified diff patch.
type UnifiedDiff struct {
	OldFile string
	NewFile string
	Hunks   []UnifiedHunk
}

var hunkHeaderRe = regexp.MustCompile(`^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@`)

// ExtractFencedBlock extracts the body of the first matching markdown code fence,
// or returns trimmed input when no fence is present.
func ExtractFencedBlock(text string, langs ...string) string {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	langSet := make(map[string]bool, len(langs))
	for _, l := range langs {
		langSet[strings.ToLower(strings.TrimSpace(l))] = true
	}

	extract := func(matchAny bool) (string, bool) {
		inBlock := false
		var buf []string
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if !inBlock {
				if strings.HasPrefix(trimmed, "```") {
					tag := strings.ToLower(strings.TrimSpace(strings.TrimPrefix(trimmed, "```")))
					if idx := strings.IndexAny(tag, " \t"); idx >= 0 {
						tag = tag[:idx]
					}
					if matchAny || len(langSet) == 0 || langSet[tag] {
						inBlock = true
						buf = buf[:0]
					}
				}
				continue
			}
			if strings.HasPrefix(trimmed, "```") {
				return strings.Join(buf, "\n"), true
			}
			buf = append(buf, line)
		}
		if inBlock && len(buf) > 0 {
			return strings.Join(buf, "\n"), true
		}
		return "", false
	}

	if len(langSet) > 0 {
		if body, ok := extract(false); ok {
			return strings.TrimSpace(body)
		}
	}
	if body, ok := extract(true); ok {
		return strings.TrimSpace(body)
	}
	return strings.TrimSpace(text)
}

// ParseUnifiedDiff parses and validates unified diff hunks and line prefixes.
// When strictCounts is true, hunk @@ -a,b +c,d @@ counts must match exact line tallies.
func ParseUnifiedDiff(raw string) (*UnifiedDiff, error) {
	return parseUnifiedDiff(raw, false)
}

// ParseUnifiedDiffStrict parses a unified diff and requires exact @@ line count tallies.
func ParseUnifiedDiffStrict(raw string) (*UnifiedDiff, error) {
	return parseUnifiedDiff(raw, true)
}

func parseUnifiedDiff(raw string, strictCounts bool) (*UnifiedDiff, error) {
	body := ExtractFencedBlock(raw, "diff", "patch", "udiff")
	if body == "" {
		return nil, errors.New("empty diff")
	}
	lines := strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n")
	diff := &UnifiedDiff{}
	var cur *UnifiedHunk
	var oldSeen, newSeen int

	flush := func() error {
		if cur == nil {
			return nil
		}
		if len(cur.Lines) == 0 {
			return fmt.Errorf("hunk @@ -%d,%d +%d,%d @@ is empty",
				cur.OldStart, cur.OldCount, cur.NewStart, cur.NewCount)
		}
		if strictCounts && (oldSeen != cur.OldCount || newSeen != cur.NewCount) {
			return fmt.Errorf("hunk @@ -%d,%d +%d,%d @@ line count mismatch: got -%d +%d",
				cur.OldStart, cur.OldCount, cur.NewStart, cur.NewCount, oldSeen, newSeen)
		}
		diff.Hunks = append(diff.Hunks, *cur)
		cur = nil
		oldSeen, newSeen = 0, 0
		return nil
	}

	for _, line := range lines {
		if strings.HasPrefix(line, "--- ") && cur == nil {
			diff.OldFile = strings.TrimSpace(strings.TrimPrefix(line, "--- "))
			continue
		}
		if strings.HasPrefix(line, "+++ ") && cur == nil {
			diff.NewFile = strings.TrimSpace(strings.TrimPrefix(line, "+++ "))
			continue
		}
		if m := hunkHeaderRe.FindStringSubmatch(line); m != nil {
			if err := flush(); err != nil {
				return nil, err
			}
			oldStart, _ := strconv.Atoi(m[1])
			oldCount := 1
			if m[2] != "" {
				oldCount, _ = strconv.Atoi(m[2])
			}
			newStart, _ := strconv.Atoi(m[3])
			newCount := 1
			if m[4] != "" {
				newCount, _ = strconv.Atoi(m[4])
			}
			cur = &UnifiedHunk{
				OldStart: oldStart,
				OldCount: oldCount,
				NewStart: newStart,
				NewCount: newCount,
			}
			continue
		}
		if cur == nil {
			continue
		}
		if line == `\ No newline at end of file` {
			continue
		}
		if line == "" {
			// Treat blank line inside hunk as empty context line.
			cur.Lines = append(cur.Lines, " ")
			oldSeen++
			newSeen++
			continue
		}
		switch line[0] {
		case ' ':
			cur.Lines = append(cur.Lines, line)
			oldSeen++
			newSeen++
		case '-':
			cur.Lines = append(cur.Lines, line)
			oldSeen++
		case '+':
			cur.Lines = append(cur.Lines, line)
			newSeen++
		default:
			if len(cur.Lines) > 0 && (!strictCounts || (oldSeen == cur.OldCount && newSeen == cur.NewCount)) {
				if err := flush(); err != nil {
					return nil, err
				}
				continue
			}
			return nil, fmt.Errorf("unexpected diff line %q inside hunk", line)
		}
	}
	if err := flush(); err != nil {
		return nil, err
	}
	if len(diff.Hunks) == 0 {
		return nil, errors.New("no valid @@ unified diff hunks found")
	}
	return diff, nil
}

// ApplyUnifiedDiff applies a validated unified diff to original file contents.
func ApplyUnifiedDiff(original, rawDiff string) (string, error) {
	diff, err := ParseUnifiedDiff(rawDiff)
	if err != nil {
		return "", err
	}
	normOrig := strings.ReplaceAll(original, "\r\n", "\n")
	hadTrailingNL := strings.HasSuffix(normOrig, "\n")
	origLines := strings.Split(strings.TrimSuffix(normOrig, "\n"), "\n")
	if normOrig == "" {
		origLines = nil
	}

	var out []string
	cursor := 0
	for _, h := range diff.Hunks {
		var oldPattern []string
		var newSegment []string
		for _, dl := range h.Lines {
			switch dl[0] {
			case ' ':
				oldPattern = append(oldPattern, dl[1:])
				newSegment = append(newSegment, dl[1:])
			case '-':
				oldPattern = append(oldPattern, dl[1:])
			case '+':
				newSegment = append(newSegment, dl[1:])
			}
		}

		targetStart := max(0, h.OldStart-1)
		matchIdx := findHunkMatch(origLines, oldPattern, cursor, targetStart)
		if matchIdx < 0 {
			return "", fmt.Errorf("hunk @@ -%d,%d +%d,%d @@ context did not match source",
				h.OldStart, h.OldCount, h.NewStart, h.NewCount)
		}
		out = append(out, origLines[cursor:matchIdx]...)
		out = append(out, newSegment...)
		cursor = matchIdx + len(oldPattern)
	}
	out = append(out, origLines[cursor:]...)
	res := strings.Join(out, "\n")
	if hadTrailingNL && res != "" {
		res += "\n"
	}
	return res, nil
}

func findHunkMatch(origLines, pattern []string, minStart, hint int) int {
	if len(pattern) == 0 {
		if hint >= minStart && hint <= len(origLines) {
			return hint
		}
		if minStart <= len(origLines) {
			return minStart
		}
		return -1
	}
	maxStart := len(origLines) - len(pattern)
	if maxStart < minStart {
		return -1
	}
	matchesAt := func(pos int) bool {
		for i, p := range pattern {
			if origLines[pos+i] != p {
				return false
			}
		}
		return true
	}
	if hint >= minStart && hint <= maxStart && matchesAt(hint) {
		return hint
	}
	// Search outward from hint while respecting sequential hunk cursor.
	for delta := 1; delta <= len(origLines); delta++ {
		if hint-delta >= minStart && hint-delta <= maxStart && matchesAt(hint-delta) {
			return hint - delta
		}
		if hint+delta >= minStart && hint+delta <= maxStart && matchesAt(hint+delta) {
			return hint + delta
		}
	}
	return -1
}

// ParseAndCollectGoSymbols parses Go source code and collects declared symbol names.
func ParseAndCollectGoSymbols(src string) (map[string]bool, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "card.go", src, parser.AllErrors)
	if err != nil {
		return nil, err
	}
	syms := make(map[string]bool)
	if file.Name != nil && file.Name.Name != "" {
		syms["package:"+file.Name.Name] = true
	}
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Name != nil {
				syms[d.Name.Name] = true
				if d.Recv != nil && len(d.Recv.List) > 0 {
					recvName := exprTypeName(d.Recv.List[0].Type)
					if recvName != "" {
						syms[recvName+"."+d.Name.Name] = true
					}
				}
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					if s.Name != nil {
						syms[s.Name.Name] = true
						if st, ok := s.Type.(*ast.StructType); ok && st.Fields != nil {
							for _, f := range st.Fields.List {
								for _, fn := range f.Names {
									syms[fn.Name] = true
									syms[s.Name.Name+"."+fn.Name] = true
								}
							}
						}
					}
				case *ast.ValueSpec:
					for _, vn := range s.Names {
						syms[vn.Name] = true
					}
				}
			}
		}
	}
	return syms, nil
}

func exprTypeName(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return exprTypeName(t.X)
	case *ast.IndexExpr:
		return exprTypeName(t.X)
	default:
		return ""
	}
}

// EvaluateCard executes every deterministic check configured on a card.
func EvaluateCard(card Card, res ChatResult) CardResult {
	out := CardResult{
		CardID:   card.ID,
		Kind:     card.Kind,
		Response: res,
	}
	add := func(name string, ok bool, detail string) {
		out.Outcomes = append(out.Outcomes, CheckOutcome{Name: name, Passed: ok, Detail: detail})
		out.ChecksTotal++
		if ok {
			out.ChecksPassed++
		}
	}

	targetText := strings.TrimSpace(res.Content)
	var parsedArgs map[string]any

	if card.Checks.RequireValidToolCall {
		ok, detail, args, canonical := verifyToolCall(card.Checks, res.ToolCalls)
		add("tool_call_json", ok, detail)
		parsedArgs = args
		if canonical != "" {
			out.NormalizedOutput = canonical
		}
		if ok && card.Checks.ToolArgDiffField != "" {
			if v, exists := parsedArgs[card.Checks.ToolArgDiffField].(string); exists {
				targetText = v
			}
		} else if ok && card.Checks.ToolArgCodeField != "" {
			if v, exists := parsedArgs[card.Checks.ToolArgCodeField].(string); exists {
				targetText = v
			}
		} else if ok && len(res.ToolCalls) > 0 {
			if oldText, okOld := parsedArgs["old_text"].(string); okOld {
				if newText, okNew := parsedArgs["new_text"].(string); okNew && card.SourceContent != "" {
					if strings.Contains(card.SourceContent, oldText) {
						targetText = strings.Replace(card.SourceContent, oldText, newText, 1)
					} else {
						targetText = newText
					}
				}
			}
		}
	}

	if out.NormalizedOutput == "" {
		out.NormalizedOutput = strings.TrimSpace(res.Content)
	}

	if card.Checks.RequireUnifiedDiff {
		_, err := ParseUnifiedDiff(targetText)
		if err != nil {
			add("unified_diff", false, err.Error())
		} else {
			add("unified_diff", true, "valid unified diff hunks")
		}
	}

	if card.Checks.ApplyDiffToSource {
		patched, err := ApplyUnifiedDiff(card.SourceContent, targetText)
		if err != nil {
			add("diff_applies", false, err.Error())
		} else {
			add("diff_applies", true, "patch applies cleanly")
			targetText = patched
		}
	}

	if card.Checks.RequireValidGoParse || len(card.Checks.ExpectedGoSymbols) > 0 {
		goSrc := ExtractFencedBlock(targetText, "go", "golang")
		syms, err := ParseAndCollectGoSymbols(goSrc)
		if err != nil {
			add("go_parse", false, err.Error())
		} else {
			add("go_parse", true, fmt.Sprintf("parsed %d symbols", len(syms)))
			for _, want := range card.Checks.ExpectedGoSymbols {
				if !syms[want] {
					add("symbol:"+want, false, fmt.Sprintf("missing Go symbol %q", want))
				} else {
					add("symbol:"+want, true, "present")
				}
			}
		}
		targetText = goSrc
	}

	for _, sub := range card.Checks.MustContain {
		ok := strings.Contains(targetText, sub)
		detail := "present"
		if !ok {
			detail = fmt.Sprintf("missing required substring %q", sub)
		}
		add("must_contain:"+sub, ok, detail)
	}

	for _, sub := range card.Checks.MustNotContain {
		ok := !strings.Contains(targetText, sub)
		detail := "absent"
		if !ok {
			detail = fmt.Sprintf("found forbidden substring %q", sub)
		}
		add("must_not_contain:"+sub, ok, detail)
	}

	for _, pattern := range card.Checks.MustMatchRegex {
		re, err := regexp.Compile(pattern)
		if err != nil {
			add("regex:"+pattern, false, fmt.Sprintf("invalid regex: %v", err))
			continue
		}
		ok := re.MatchString(targetText)
		detail := "matched"
		if !ok {
			detail = fmt.Sprintf("did not match regex %q", pattern)
		}
		add("regex:"+pattern, ok, detail)
	}

	out.Passed = out.ChecksTotal > 0 && out.ChecksPassed == out.ChecksTotal
	return out
}

func verifyToolCall(spec CheckSpec, calls []ToolCall) (bool, string, map[string]any, string) {
	if len(calls) == 0 {
		return false, "expected tool_calls, got none", nil, ""
	}
	call := calls[0]
	if spec.ExpectedToolName != "" && call.Function.Name != spec.ExpectedToolName {
		return false, fmt.Sprintf("expected tool %q, got %q", spec.ExpectedToolName, call.Function.Name), nil, ""
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return false, fmt.Sprintf("invalid tool_call JSON arguments: %v", err), nil, ""
	}
	if args == nil {
		return false, "tool_call arguments must be a JSON object", nil, ""
	}
	for _, reqKey := range spec.RequiredToolArgs {
		val, exists := args[reqKey]
		if !exists || val == nil {
			return false, fmt.Sprintf("missing required tool argument %q", reqKey), args, ""
		}
		if s, isStr := val.(string); isStr && strings.TrimSpace(s) == "" {
			return false, fmt.Sprintf("required tool argument %q is empty", reqKey), args, ""
		}
	}
	for k, want := range spec.ToolArgEquals {
		got, exists := args[k]
		if !exists {
			return false, fmt.Sprintf("missing tool argument %q", k), args, ""
		}
		if fmt.Sprint(got) != want {
			return false, fmt.Sprintf("tool argument %q = %v, want %q", k, got, want), args, ""
		}
	}
	canonBytes, _ := json.Marshal(struct {
		Name string         `json:"name"`
		Args map[string]any `json:"args"`
	}{Name: call.Function.Name, Args: args})
	return true, "valid tool call JSON", args, string(canonBytes)
}
