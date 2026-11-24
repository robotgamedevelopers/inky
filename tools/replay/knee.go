package main

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// RungMetrics summarizes one concurrency rung of a replay sweep.
type RungMetrics struct {
	Lanes           int     `json:"lanes"`
	TotalTokS       float64 `json:"total_tok_s"`
	PerLaneTokS     float64 `json:"per_lane_tok_s"`
	TTFTP50Ms       float64 `json:"ttft_p50_ms"`
	TTFTP95Ms       float64 `json:"ttft_p95_ms"`
	PeakMemoryBytes uint64  `json:"peak_memory_bytes"`
	CardsCompleted  int     `json:"cards_completed"`
	CardsPerHour    float64 `json:"cards_per_hour"`
	Errors          int     `json:"errors"`
	Retries429      int     `json:"retries_429"`
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

	var completed []RungMetrics
	for _, r := range sorted {
		if r.CardsCompleted > 0 {
			completed = append(completed, r)
		}
	}
	if len(completed) == 0 {
		return KneeResult{}, errors.New("no completed cards across any concurrency rung")
	}

	firstIdx := -1
	for i, r := range completed {
		if ok, _ := withinSLO(r, cfg); ok {
			firstIdx = i
			break
		}
	}
	if firstIdx == -1 {
		best := completed[0]
		for _, r := range completed[1:] {
			if r.Errors < best.Errors || (r.Errors == best.Errors && r.CardsPerHour > best.CardsPerHour) {
				best = r
			}
		}
		return KneeResult{
			RecommendedLanes: best.Lanes,
			Reason:           "fallback to lowest-error / highest-throughput rung (no clean SLO-compliant rung)",
			Rung:             best,
		}, nil
	}

	best := completed[firstIdx]
	for _, curr := range completed[firstIdx+1:] {
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

// FormatMarkdown renders a Markdown summary table and knee recommendation.
func FormatMarkdown(rungs []RungMetrics, knee KneeResult) string {
	sorted := make([]RungMetrics, len(rungs))
	copy(sorted, rungs)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Lanes < sorted[j].Lanes
	})

	var b strings.Builder
	b.WriteString("| lanes | total_tok_s | per_lane_tok_s | ttft_p50_ms | ttft_p95_ms | peak_memory_bytes | cards_completed | cards_per_hour | errors | retries_429 |\n")
	b.WriteString("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range sorted {
		fmt.Fprintf(&b, "| %d | %.2f | %.2f | %.1f | %.1f | %d | %d | %.1f | %d | %d |\n",
			r.Lanes, r.TotalTokS, r.PerLaneTokS, r.TTFTP50Ms, r.TTFTP95Ms,
			r.PeakMemoryBytes, r.CardsCompleted, r.CardsPerHour, r.Errors, r.Retries429)
	}
	fmt.Fprintf(&b, "\nRecommended default lane count: **%d** (%s)\n", knee.RecommendedLanes, knee.Reason)
	return b.String()
}

// FormatTSV renders a tab-separated summary of all rungs.
func FormatTSV(rungs []RungMetrics) string {
	sorted := make([]RungMetrics, len(rungs))
	copy(sorted, rungs)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Lanes < sorted[j].Lanes
	})

	var b strings.Builder
	b.WriteString("lanes\ttotal_tok_s\tper_lane_tok_s\tttft_p50_ms\tttft_p95_ms\tpeak_memory_bytes\tcards_completed\tcards_per_hour\terrors\tretries_429\n")
	for _, r := range sorted {
		fmt.Fprintf(&b, "%d\t%.2f\t%.2f\t%.1f\t%.1f\t%d\t%d\t%.1f\t%d\t%d\n",
			r.Lanes, r.TotalTokS, r.PerLaneTokS, r.TTFTP50Ms, r.TTFTP95Ms,
			r.PeakMemoryBytes, r.CardsCompleted, r.CardsPerHour, r.Errors, r.Retries429)
	}
	return b.String()
}
