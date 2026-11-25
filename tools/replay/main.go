// replay: drives recorded multi-turn card traffic across lane concurrencies and selects the throughput knee.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

func parseLaneList(raw string) ([]int, error) {
	parts := strings.Split(raw, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil || n < 1 {
			return nil, fmt.Errorf("invalid lane count %q: must be positive integer", p)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("--lanes must specify at least one lane count")
	}
	return out, nil
}

func runCLI(args []string, getenv func(string) string, stdout, stderr io.Writer) int {
	return runCLIWithDeps(args, getenv, nil, nil, stdout, stderr)
}

func runCLIWithDeps(args []string, getenv func(string) string, client *http.Client, clk Clock, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("replay", flag.ContinueOnError)
	fs.SetOutput(stderr)
	url := fs.String("url", "http://127.0.0.1:8080", "base URL of inky serve")
	input := fs.String("input", "", "path to recorded .jsonl file or directory with .jsonl and .jsonl.1 segments")
	lanesRaw := fs.String("lanes", "8,16,32,48,64,96,128", "comma-separated lane counts to sweep")
	cardsPerRung := fs.Int("cards", 0, "cards to run per rung (0 = max(loaded cards, lanes))")
	duration := fs.Duration("duration", 0, "optional fixed duration per rung (e.g. 30s; 0 uses --cards)")
	maxRetries := fs.Int("max-retries", 8, "max 429 retries per turn")
	healthInterval := fs.Duration("health-interval", 250*time.Millisecond, "/health polling interval")
	serverPID := fs.Int("server-pid", 0, "optional server PID to sample OS RSS via ps")
	measureHeadroom := fs.Bool("measure-headroom", false, "sample host memory headroom via vm_stat/sysctl or /proc/meminfo")
	forceStream := fs.Bool("force-stream", false, "override stream=true on replayed requests")
	minGain := fs.Float64("min-gain", 0.05, "minimum marginal throughput gain per doubling for knee selection")
	maxTTFTP95Ms := fs.Float64("max-ttft-p95-ms", 0, "optional p95 TTFT ceiling in ms (0 = disabled)")
	minPerLaneTokS := fs.Float64("min-per-lane-tok-s", 0, "optional per-lane tok/s floor (0 = disabled)")
	keyEnv := fs.String("key-env", "", "optional environment variable holding bearer API key")
	outTSV := fs.String("out-tsv", "", "optional path to write TSV summary")
	outMD := fs.String("out-md", "", "optional path to write Markdown summary")

	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *duration < 0 {
		fmt.Fprintln(stderr, "--duration must be nonnegative")
		return 2
	}
	if *serverPID < 0 {
		fmt.Fprintln(stderr, "--server-pid must be nonnegative")
		return 2
	}
	if strings.TrimSpace(*input) == "" {
		fmt.Fprintln(stderr, "--input is required")
		return 2
	}
	lanes, err := parseLaneList(*lanesRaw)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	cards, err := LoadReplaySet(*input)
	if err != nil {
		fmt.Fprintf(stderr, "load replay set: %v\n", err)
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

	cfg := SweepConfig{
		BaseURL:         *url,
		APIKey:          apiKey,
		LaneCounts:      lanes,
		CardsPerRung:    *cardsPerRung,
		Duration:        *duration,
		MaxRetries:      *maxRetries,
		HealthInterval:  *healthInterval,
		ServerPID:       *serverPID,
		MeasureHeadroom: *measureHeadroom,
		ForceStream:     *forceStream,
		Knee: KneeConfig{
			MinMarginalGain: *minGain,
			MaxTTFTP95Ms:    *maxTTFTP95Ms,
			MinPerLaneTokS:  *minPerLaneTokS,
		},
		Client: client,
		Clock:  clk,
	}

	rungs, knee, err := RunSweep(context.Background(), cfg, cards)
	if err != nil {
		fmt.Fprintf(stderr, "sweep failed: %v\n", err)
		if len(rungs) == 0 {
			return 1
		}
	}

	md := FormatMarkdown(rungs, knee)
	tsv := FormatTSV(rungs)
	fmt.Fprint(stdout, md)

	if *outMD != "" {
		if err := os.WriteFile(*outMD, []byte(md), 0o644); err != nil {
			fmt.Fprintf(stderr, "write --out-md: %v\n", err)
			return 1
		}
	}
	if *outTSV != "" {
		if err := os.WriteFile(*outTSV, []byte(tsv), 0o644); err != nil {
			fmt.Fprintf(stderr, "write --out-tsv: %v\n", err)
			return 1
		}
	}
	if err != nil {
		return 1
	}
	return 0
}

func main() {
	os.Exit(runCLI(os.Args[1:], os.Getenv, os.Stdout, os.Stderr))
}
