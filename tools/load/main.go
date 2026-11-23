// load: N concurrent chat completions against an OpenAI-compatible endpoint; prints total and per-stream decode rates.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"sort"
	"sync"
	"time"
)

var topics = []string{"a ring buffer", "a hash map with open addressing", "a binary heap", "a trie", "an LRU cache", "a bloom filter",
	"a skip list", "a lock-free queue", "a B-tree", "a union-find structure", "a rope", "a segment tree", "a bitset", "a slab allocator",
	"an interval tree", "a radix sort"}
var langs = []string{"C", "Go", "Rust", "Zig"}

func main() {
	url := flag.String("url", "http://127.0.0.1:8091/v1/chat/completions", "")
	n := flag.Int("n", 1, "concurrent streams")
	predict := flag.Int("predict", 256, "tokens per stream")
	pad := flag.Int("pad", 0, "repeat a filler paragraph this many times in each prompt (context size)")
	flag.Parse()
	type res struct {
		toks int
		rate float64
		err  error
	}
	out := make([]res, *n)
	var wg sync.WaitGroup
	start := time.Now()
	for i := 0; i < *n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			p := fmt.Sprintf("Explain how to implement %s in %s, with code, then list its failure modes.", topics[i%len(topics)], langs[(i/len(topics))%len(langs)])
			for k := 0; k < *pad; k++ {
				p = fmt.Sprintf("Background note %d-%d: the system under discussion handles requests in a loop, validates each input, and logs every refusal with its reason.\n", i, k) + p
			}
			body, _ := json.Marshal(map[string]any{"model": "inkling", "max_tokens": *predict, "temperature": 0.7, "reasoning_effort": "none",
				"messages": []map[string]string{{"role": "user", "content": p}}})
			r, err := http.Post(*url, "application/json", bytes.NewReader(body))
			if err != nil {
				out[i].err = err
				return
			}
			defer r.Body.Close()
			b, _ := io.ReadAll(r.Body)
			var v struct {
				Usage   struct{ CompletionTokens int `json:"completion_tokens"` } `json:"usage"`
				Timings struct{ PredictedPerSecond float64 `json:"predicted_per_second"` } `json:"timings"`
			}
			if err := json.Unmarshal(b, &v); err != nil || v.Usage.CompletionTokens == 0 {
				out[i].err = fmt.Errorf("bad response: %.200s", b)
				return
			}
			out[i] = res{toks: v.Usage.CompletionTokens, rate: v.Timings.PredictedPerSecond}
		}(i)
	}
	wg.Wait()
	wall := time.Since(start).Seconds()
	total, errs := 0, 0
	var rates []float64
	for _, r := range out {
		if r.err != nil {
			errs++
			if errs == 1 {
				fmt.Println("error:", r.err)
			}
			continue
		}
		total += r.toks
		rates = append(rates, r.rate)
	}
	sort.Float64s(rates)
	med := 0.0
	if len(rates) > 0 {
		med = rates[len(rates)/2]
	}
	fmt.Printf("n=%d tokens=%d wall=%.1fs total=%.0f tok/s per-stream-median=%.1f tok/s errors=%d\n", *n, total, wall, float64(total)/wall, med, errs)
}
