// quality: deterministic coding/patch/JSON-tool-call replay evaluator for fp16 vs --kv8 and reasoning_effort sweeps.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

func parseCSV(raw string) []string {
	var out []string
	for _, p := range strings.Split(raw, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func runCLI(args []string, getenv func(string) string, client *http.Client, clk Clock, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("quality", flag.ContinueOnError)
	fs.SetOutput(stderr)
	cardsPath := fs.String("cards", "tests/quality/cards.jsonl", "path to benchmark cards JSONL")
	url := fs.String("url", "http://127.0.0.1:8090", "base URL of inky serve")
	refURL := fs.String("ref-url", "", "optional reference fp16 server URL for live side-by-side comparison")
	kv8URL := fs.String("kv8-url", "", "optional --kv8 server URL for live side-by-side comparison")
	modeName := fs.String("name", "", "run label (e.g. fp16-medium)")
	effortsRaw := fs.String("efforts", "medium", "comma-separated reasoning_effort levels to run")
	temp := fs.Float64("temp", 0.0, "sampling temperature")
	seed := fs.Int64("seed", 42, "base deterministic seed")
	maxTokens := fs.Int("max-tokens", 0, "optional global max_tokens override (0 = per-card default)")
	maxRetries := fs.Int("max-retries", 4, "max HTTP 429 retries per card")
	keyEnv := fs.String("key-env", "", "optional environment variable holding bearer API key")
	hardware := fs.String("hardware", "", "optional hardware description for report metadata")
	model := fs.String("model", "inkling-2-mini", "model field in request payload")
	outJSON := fs.String("out-json", "", "optional path to write JSON run or full report")
	outMD := fs.String("out-md", "", "optional path to write Markdown summary")
	mergeRaw := fs.String("merge", "", "comma-separated ModeRun JSON files to combine into a FullReport without hitting a server")
	refRunName := fs.String("ref-run", "fp16-medium", "run name to use as fp16 reference when merging")
	kv8RunName := fs.String("kv8-run", "kv8-medium", "run name to use as kv8 candidate when merging")

	if err := fs.Parse(args); err != nil {
		return 2
	}

	var apiKey string
	if *keyEnv != "" {
		apiKey = getenv(*keyEnv)
		if apiKey == "" {
			fmt.Fprintf(stderr, "--key-env %s is unset or empty\n", *keyEnv)
			return 2
		}
	}

	if strings.TrimSpace(*mergeRaw) != "" {
		return runMerge(parseCSV(*mergeRaw), *refRunName, *kv8RunName, *hardware, *model, *outJSON, *outMD, stdout, stderr)
	}

	cards, err := LoadCards(*cardsPath)
	if err != nil {
		fmt.Fprintf(stderr, "load cards: %v\n", err)
		return 2
	}

	ctx := context.Background()
	efforts := parseCSV(*effortsRaw)
	if len(efforts) == 0 {
		fmt.Fprintln(stderr, "--efforts must contain at least one level")
		return 2
	}
	if *kv8URL != "" && len(efforts) > 1 {
		fmt.Fprintln(stderr, "--kv8-url cannot be combined with multiple --efforts levels")
		return 2
	}

	var runs []ModeRun
	var cmp *ModeComparison
	if *refURL != "" && *kv8URL != "" {
		eff := efforts[0]
		refRun, err := RunSuite(ctx, RunConfig{
			Name: "fp16-" + eff, BaseURL: *refURL, APIKey: apiKey, Model: *model,
			ReasoningEffort: eff, Temperature: *temp, Seed: *seed, MaxTokens: *maxTokens,
			MaxRetries: *maxRetries, Client: client, Clock: clk, Progress: stderr,
		}, cards)
		if err != nil {
			fmt.Fprintf(stderr, "reference run failed: %v\n", err)
			return 1
		}
		kv8Run, err := RunSuite(ctx, RunConfig{
			Name: "kv8-" + eff, BaseURL: *kv8URL, APIKey: apiKey, Model: *model,
			ReasoningEffort: eff, Temperature: *temp, Seed: *seed, MaxTokens: *maxTokens,
			MaxRetries: *maxRetries, Client: client, Clock: clk, Progress: stderr,
		}, cards)
		if err != nil {
			fmt.Fprintf(stderr, "kv8 run failed: %v\n", err)
			return 1
		}
		c, err := CompareModes(refRun, kv8Run)
		if err != nil {
			fmt.Fprintf(stderr, "compare modes: %v\n", err)
			return 1
		}
		cmp = &c
		runs = append(runs, refRun, kv8Run)
	} else {
		for _, eff := range efforts {
			prefix := *modeName
			if prefix == "" {
				prefix = "run"
			}
			name := prefix
			if !strings.HasSuffix(name, "-"+eff) {
				name = prefix + "-" + eff
			}
			run, err := RunSuite(ctx, RunConfig{
				Name: name, BaseURL: *url, APIKey: apiKey, Model: *model,
				ReasoningEffort: eff, Temperature: *temp, Seed: *seed, MaxTokens: *maxTokens,
				MaxRetries: *maxRetries, Client: client, Clock: clk, Progress: stderr,
			}, cards)
			if err != nil {
				fmt.Fprintf(stderr, "run (%s) failed: %v\n", eff, err)
				return 1
			}
			runs = append(runs, run)
		}
	}

	var sweep *EffortSweepReport
	if len(efforts) > 1 && *kv8URL == "" {
		s := BuildEffortSweep(runs, efforts)
		sweep = &s
	}

	report := FullReport{
		Hardware:    *hardware,
		Model:       *model,
		Comparison:  cmp,
		EffortSweep: sweep,
		Runs:        runs,
	}
	return writeOutputs(report, *outJSON, *outMD, stdout, stderr)
}

func runMerge(files []string, refName, kv8Name, hardware, model, outJSON, outMD string, stdout, stderr io.Writer) int {
	var runs []ModeRun
	for _, path := range files {
		data, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(stderr, "read %s: %v\n", path, err)
			return 1
		}
		var rep FullReport
		if err := json.Unmarshal(data, &rep); err == nil && len(rep.Runs) > 0 {
			runs = append(runs, rep.Runs...)
			continue
		}
		var single ModeRun
		if err := json.Unmarshal(data, &single); err != nil || len(single.Results) == 0 {
			fmt.Fprintf(stderr, "decode %s as FullReport or ModeRun failed\n", path)
			return 1
		}
		runs = append(runs, single)
	}
	byName := make(map[string]ModeRun, len(runs))
	var fp16Runs []ModeRun
	for _, r := range runs {
		byName[r.Name] = r
		if r.KVCacheMode == "fp16" {
			fp16Runs = append(fp16Runs, r)
		}
	}
	var cmp *ModeComparison
	if ref, okRef := byName[refName]; okRef {
		if cand, okCand := byName[kv8Name]; okCand {
			c, err := CompareModes(ref, cand)
			if err != nil {
				fmt.Fprintf(stderr, "compare %s vs %s: %v\n", refName, kv8Name, err)
				return 1
			}
			cmp = &c
		}
	}
	var sweep *EffortSweepReport
	if len(fp16Runs) > 1 {
		s := BuildEffortSweep(fp16Runs, []string{"none", "low", "medium", "high"})
		sweep = &s
	}
	report := FullReport{
		Hardware:    hardware,
		Model:       model,
		Comparison:  cmp,
		EffortSweep: sweep,
		Runs:        runs,
	}
	return writeOutputs(report, outJSON, outMD, stdout, stderr)
}

func writeOutputs(report FullReport, outJSON, outMD string, stdout, stderr io.Writer) int {
	md := FormatMarkdown(report)
	if md != "" {
		fmt.Fprint(stdout, md)
	} else {
		for _, r := range report.Runs {
			fmt.Fprintf(stdout, "%s (%s/%s): %d/%d passed (%.1f%%), avg %.1f ms, %d gen tokens\n",
				r.Name, r.KVCacheMode, r.ReasoningEffort, r.CardsPassed, r.CardsTotal, r.PassRate*100, r.AvgWallMs, r.TotalCompTokens)
		}
	}
	if outJSON != "" {
		raw, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "marshal json: %v\n", err)
			return 1
		}
		if err := os.WriteFile(outJSON, append(raw, '\n'), 0o644); err != nil {
			fmt.Fprintf(stderr, "write --out-json: %v\n", err)
			return 1
		}
	}
	if outMD != "" && md != "" {
		if err := os.WriteFile(outMD, []byte(md), 0o644); err != nil {
			fmt.Fprintf(stderr, "write --out-md: %v\n", err)
			return 1
		}
	}
	return 0
}

func main() {
	os.Exit(runCLI(os.Args[1:], os.Getenv, nil, nil, os.Stdout, os.Stderr))
}
