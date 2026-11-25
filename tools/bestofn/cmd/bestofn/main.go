package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"bestofn"
)

func main() {
	cfg := bestofn.DefaultConfig()
	var (
		apiKeyEnv  string
		prompt     string
		promptFile string
		targetFile string
		cardPath   string
		cardID     string
		tier       string
		noStream   bool
		noCancel   bool
		jsonOut    bool
	)

	flag.StringVar(&cfg.Endpoint, "endpoint", cfg.Endpoint, "OpenAI-compatible server base URL")
	flag.StringVar(&apiKeyEnv, "api-key-env", "", "optional environment variable holding Bearer API key")
	flag.StringVar(&cfg.Model, "model", cfg.Model, "model name sent to /v1/chat/completions")
	flag.IntVar(&cfg.N, "n", cfg.N, "parallel attempts per card")
	flag.Float64Var(&cfg.Temperature, "temp", cfg.Temperature, "sampling temperature")
	flag.IntVar(&cfg.MaxTokens, "max-tokens", cfg.MaxTokens, "max completion tokens per lane")
	flag.Uint64Var(&cfg.BaseSeed, "seed", cfg.BaseSeed, "base seed (lane i uses seed+i)")
	flag.DurationVar(&cfg.Timeout, "timeout", cfg.Timeout, "per-card wall-clock timeout")
	flag.DurationVar(&cfg.CheckTimeout, "check-timeout", cfg.CheckTimeout, "per-attempt check timeout")
	flag.StringVar(&cfg.SystemPrompt, "system", "", "shared system prompt across all lanes")
	flag.StringVar(&prompt, "prompt", "", "single-card user prompt")
	flag.StringVar(&promptFile, "prompt-file", "", "read single-card prompt from file")
	flag.StringVar(&targetFile, "target-file", "", "filename inside temp dir to write extracted code block")
	flag.StringVar(&cfg.CheckCmd, "check", "", "deterministic shell command to verify each candidate")
	flag.StringVar(&cfg.WorkDir, "workdir", "", "optional directory copied into each lane's isolated temp dir")
	flag.StringVar(&cardPath, "card", "", "JSON file containing one Card or an array of Cards")
	flag.StringVar(&cardID, "card-id", "card-1", "identifier for a single-card CLI run")
	flag.StringVar(&tier, "tier", "default", "tier label for a single-card CLI run")
	flag.BoolVar(&noStream, "no-stream", false, "disable SSE streaming (not recommended for early cancellation)")
	flag.BoolVar(&noCancel, "no-cancel", false, "do not cancel remaining lanes after the first pass")
	flag.BoolVar(&jsonOut, "json", false, "emit full JSON report to stdout")
	flag.Parse()

	cfg.Stream = !noStream
	cfg.CancelOnFirstPass = !noCancel
	if apiKeyEnv != "" {
		cfg.APIKey = os.Getenv(apiKeyEnv)
	}

	var cards []bestofn.Card
	if cardPath != "" {
		data, err := os.ReadFile(cardPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "bestofn: read %s: %v\n", cardPath, err)
			os.Exit(2)
		}
		trimmed := strings.TrimSpace(string(data))
		if strings.HasPrefix(trimmed, "[") {
			if err := json.Unmarshal(data, &cards); err != nil {
				fmt.Fprintf(os.Stderr, "bestofn: parse card array %s: %v\n", cardPath, err)
				os.Exit(2)
			}
		} else {
			var single bestofn.Card
			if err := json.Unmarshal(data, &single); err != nil {
				fmt.Fprintf(os.Stderr, "bestofn: parse card %s: %v\n", cardPath, err)
				os.Exit(2)
			}
			cards = []bestofn.Card{single}
		}
	} else {
		if promptFile != "" {
			b, err := os.ReadFile(promptFile)
			if err != nil {
				fmt.Fprintf(os.Stderr, "bestofn: read prompt file %s: %v\n", promptFile, err)
				os.Exit(2)
			}
			prompt = string(b)
		}
		cards = []bestofn.Card{{
			ID:         cardID,
			Tier:       tier,
			Prompt:     prompt,
			TargetFile: targetFile,
			CheckCmd:   cfg.CheckCmd,
		}}
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(len(cards))*cfg.Timeout+30*time.Second)
	defer cancel()

	rep, err := bestofn.RunSuite(ctx, cfg, cards)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bestofn: %v\n", err)
		os.Exit(1)
	}

	if jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(rep)
	} else {
		fmt.Print(bestofn.FormatSummary(rep))
	}
	if rep.BestOfNCount == 0 {
		os.Exit(1)
	}
}
