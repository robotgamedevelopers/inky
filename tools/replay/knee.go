package main

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// RungMetrics summarizes one concurrency rung of a replay sweep.
type RungMetrics struct {
	Lanes                   int     `json:"lanes"`
	TotalTokS               float64 `json:"total_tok_s"`
	PerLaneTokS             float64 `json:"per_lane_tok_s"`
	TTFTP50Ms               float64 `json:"ttft_p50_ms"`
	TTFTP95Ms               float64 `json:"ttft_p95_ms"`
	TTFTSamples             int     `json:"ttft_samples"`
	PeakBudgetEstimateBytes uint64  `json:"peak_budget_estimate_bytes"`
	PeakMemoryBytes         uint64  `json:"peak_memory_bytes"`
	PeakRSSBytes            uint64  `json:"peak_rss_bytes"`
	RSSSamples              int     `json:"rss_samples"`
	MinHeadroomBytes        uint64  `json:"min_headroom_bytes"`
	HostMemoryBytes         uint64  `json:"host_memory_bytes"`
	HeadroomSamples         int     `json:"headroom_samples"`
	CardsCompleted          int     `json:"cards_completed"`
	CardsPerHour            float64 `json:"cards_per_hour"`
	Errors                  int     `json:"errors"`
	Retries429              int     `json:"retries_429"`
}

// KneeConfig controls deterministic lane-count knee selection.
type KneeConfig struct {
	MinMarginalGain float64 // minimum relative gain per doubling (default 0.05 = 5%)
	MaxTTFTP95Ms    float64 // optional p95 TTFT ceiling in ms (0 = disabled)
	MinPerLaneTokS  float64 // optional per-lane tok/s floor (0 = disabled)
}

// DefaultKneeConfig returns the default knee selection thresholds.
func DefaultKneeConfig() KneeConfig {
	return KneeConfig{
		MinMarginalGain: 0.05,
	}
}

// KneeResult reports the recommended default lane count and the reason it was chosen.
type KneeResult struct {
	RecommendedLanes int         `json:"recommended_lanes"`
	Reason           string      `json:"reason"`
	Rung             RungMetrics `json:"rung"`
}

func safeRelGain(prev, curr float64) float64 {
	if prev <= 0 {
		if curr > 0 {
			return 1.0
		}
		return 0.0
	}
	return (curr - prev) / prev
}

func withinSLO(r RungMetrics, cfg KneeConfig) (bool, string) {
	if r.CardsCompleted <= 0 {
		return false, "no completed cards"
	}
	if r.Errors > 0 {
		return false, fmt.Sprintf("%d error(s) observed", r.Errors)
	}
	if cfg.MaxTTFTP95Ms > 0 && r.TTFTSamples == 0 {
		return false, "no streaming TTFT samples"
	}
	if cfg.MaxTTFTP95Ms > 0 && r.TTFTP95Ms > cfg.MaxTTFTP95Ms {
		return false, fmt.Sprintf("p95 TTFT %.1fms > %.1fms ceiling", r.TTFTP95Ms, cfg.MaxTTFTP95Ms)
	}
	if cfg.MinPerLaneTokS > 0 && r.PerLaneTokS < cfg.MinPerLaneTokS {
		return false, fmt.Sprintf("per-lane %.1f tok/s < %.1f floor", r.PerLaneTokS, cfg.MinPerLaneTokS)
	}
	return true, ""
}

// PickKnee selects the lane count where adding more lanes stops paying off
// (marginal cards/hr and total tok/s gain per doubling drops below threshold,
// errors appear, or latency/per-lane floors are crossed).
func PickKnee(rungs []RungMetrics, cfg KneeConfig) (KneeResult, error) {
	if len(rungs) == 0 {
		return KneeResult{}, errors.New("no concurrency rungs to evaluate")
	}
	if cfg.MinMarginalGain <= 0 {
		cfg.MinMarginalGain = 0.05
	}
	sorted := make([]RungMetrics, len(rungs))
	copy(sorted, rungs)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Lanes < sorted[j].Lanes
	})

	best := sorted[0]
	if ok, why := withinSLO(best, cfg); !ok {
		return KneeResult{}, fmt.Errorf("no qualifying lane count before %d lanes (%s)", best.Lanes, why)
	}
	for _, curr := range sorted[1:] {
		if ok, why := withinSLO(curr, cfg); !ok {
			return KneeResult{
				RecommendedLanes: best.Lanes,
				Reason:           fmt.Sprintf("stopped before %d lanes (%s)", curr.Lanes, why),
				Rung:             best,
			}, nil
		}
		if curr.Lanes <= best.Lanes {
			continue
		}
		doublings := math.Log2(float64(curr.Lanes) / float64(best.Lanes))
		if doublings <= 0 {
			continue
		}
		cphGain := safeRelGain(best.CardsPerHour, curr.CardsPerHour)
		tokGain := safeRelGain(best.TotalTokS, curr.TotalTokS)
		marginalPerDoubling := math.Max(cphGain, tokGain) / doublings
		if marginalPerDoubling < cfg.MinMarginalGain {
			return KneeResult{
				RecommendedLanes: best.Lanes,
				Reason: fmt.Sprintf(
					"marginal gain %.1f%%/doubling from %d to %d lanes is below %.1f%% threshold",
					marginalPerDoubling*100.0, best.Lanes, curr.Lanes, cfg.MinMarginalGain*100.0,
				),
				Rung: best,
			}, nil
		}
		best = curr
	}

	return KneeResult{
		RecommendedLanes: best.Lanes,
		Reason:           "throughput still scaling above threshold at highest tested lane count",
		Rung:             best,
	}, nil
}

// summaryRows keeps the Markdown and TSV representations of missing measurements identical.
func summaryRows(rungs []RungMetrics) [][]string {
	sorted := append([]RungMetrics(nil), rungs...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Lanes < sorted[j].Lanes })
	rows := [][]string{{"lanes", "total_tok_s", "per_lane_tok_s", "ttft_p50_ms", "ttft_p95_ms", "peak_memory_bytes", "cards_completed", "cards_per_hour", "errors", "retries_429", "ttft_samples", "peak_budget_estimate_bytes", "peak_rss_bytes", "min_headroom_bytes", "host_memory_bytes"}}
	for _, r := range sorted {
		p50, p95, memory, budget, rss, headroom, hostMem := "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"
		if r.TTFTSamples > 0 {
			p50, p95 = fmt.Sprintf("%.1f", r.TTFTP50Ms), fmt.Sprintf("%.1f", r.TTFTP95Ms)
		}
		if r.PeakMemoryBytes > 0 {
			memory = strconv.FormatUint(r.PeakMemoryBytes, 10)
		}
		if r.PeakBudgetEstimateBytes > 0 {
			budget = strconv.FormatUint(r.PeakBudgetEstimateBytes, 10)
		}
		if r.RSSSamples > 0 {
			rss = strconv.FormatUint(r.PeakRSSBytes, 10)
		}
		if r.HeadroomSamples > 0 {
			headroom = strconv.FormatUint(r.MinHeadroomBytes, 10)
			hostMem = strconv.FormatUint(r.HostMemoryBytes, 10)
		}
		rows = append(rows, []string{
			strconv.Itoa(r.Lanes), fmt.Sprintf("%.2f", r.TotalTokS), fmt.Sprintf("%.2f", r.PerLaneTokS),
			p50, p95, memory, strconv.Itoa(r.CardsCompleted), fmt.Sprintf("%.1f", r.CardsPerHour),
			strconv.Itoa(r.Errors), strconv.Itoa(r.Retries429), strconv.Itoa(r.TTFTSamples), budget,
			rss, headroom, hostMem,
		})
	}
	return rows
}

// FormatMarkdown renders a Markdown summary table and knee recommendation.
func FormatMarkdown(rungs []RungMetrics, knee KneeResult) string {
	var b strings.Builder
	for i, row := range summaryRows(rungs) {
		fmt.Fprintf(&b, "| %s |\n", strings.Join(row, " | "))
		if i == 0 {
			b.WriteString("|" + strings.Repeat("---:|", len(row)) + "\n")
		}
	}
	if knee.RecommendedLanes > 0 {
		fmt.Fprintf(&b, "\nRecommended default lane count: **%d** (%s)\n", knee.RecommendedLanes, knee.Reason)
	}
	return b.String()
}

// FormatTSV renders a tab-separated summary of all rungs.
func FormatTSV(rungs []RungMetrics) string {
	var b strings.Builder
	for _, row := range summaryRows(rungs) {
		b.WriteString(strings.Join(row, "\t") + "\n")
	}
	return b.String()
}
