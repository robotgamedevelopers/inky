// load: N concurrent chat completions; rates include prompt processing and response latency.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"
)

var topics = []string{"a ring buffer", "a hash map with open addressing", "a binary heap", "a trie", "an LRU cache", "a bloom filter",
	"a skip list", "a lock-free queue", "a B-tree", "a union-find structure", "a rope", "a segment tree", "a bitset", "a slab allocator",
	"an interval tree", "a radix sort"}
var langs = []string{"C", "Go", "Rust", "Zig"}

func runLoad(args []string, client *http.Client, now func() time.Time, stdout, stderr io.Writer) int {
	if client == nil {
		client = http.DefaultClient
	}
	if now == nil {
		now = time.Now
	}
	fs := flag.NewFlagSet("load", flag.ContinueOnError)
	fs.SetOutput(stderr)
	url := fs.String("url", "http://127.0.0.1:8091/v1/chat/completions", "")
	n := fs.Int("n", 1, "concurrent streams")
	predict := fs.Int("predict", 256, "tokens per stream")
	pad := fs.Int("pad", 0, "repeat a filler paragraph this many times in each prompt (context size)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *n < 1 || *predict < 1 || *pad < 0 {
		fmt.Fprintln(stderr, "n and predict must be positive; pad must be nonnegative")
		return 2
	}
	type res struct {
		toks int
		rate float64
		err  error
	}
	out := make([]res, *n)
	var wg sync.WaitGroup
	start := now()
	for i := 0; i < *n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			p := fmt.Sprintf("Explain how to implement %s in %s, with code, then list its failure modes.", topics[i%len(topics)], langs[(i/len(topics))%len(langs)])
			for k := 0; k < *pad; k++ {
				p = fmt.Sprintf("Background note %d-%d: the system under discussion handles requests in a loop, validates each input, and logs every refusal with its reason.\n", i, k) + p
			}
			body, _ := json.Marshal(map[string]any{"model": "inky", "max_tokens": *predict, "temperature": 0.7, "reasoning_effort": "none",
				"messages": []map[string]string{{"role": "user", "content": p}}})
			requestStart := now()
			r, err := client.Post(*url, "application/json", bytes.NewReader(body))
			if err != nil {
				out[i].err = err
				return
			}
			defer r.Body.Close()
			b, err := io.ReadAll(r.Body)
			elapsed := now().Sub(requestStart).Seconds()
			if err != nil {
				out[i].err = err
				return
			}
			if r.StatusCode != http.StatusOK {
				out[i].err = fmt.Errorf("HTTP %d: %.200s", r.StatusCode, b)
				return
			}
			var v struct {
				Usage struct {
					CompletionTokens int `json:"completion_tokens"`
				} `json:"usage"`
				Timings struct {
					PredictedPerSecond float64 `json:"predicted_per_second"`
				} `json:"timings"`
			}
			if err := json.Unmarshal(b, &v); err != nil || v.Usage.CompletionTokens == 0 {
				out[i].err = fmt.Errorf("bad response: %.200s", b)
				return
			}
			rate := v.Timings.PredictedPerSecond
			if rate <= 0 && elapsed > 0 {
				rate = float64(v.Usage.CompletionTokens) / elapsed
			}
			out[i] = res{toks: v.Usage.CompletionTokens, rate: rate}
		}(i)
	}
	wg.Wait()
	wall := now().Sub(start).Seconds()
	if wall <= 0 {
		wall = 1e-9
	}
	total, errs := 0, 0
	var rates []float64
	for _, r := range out {
		if r.err != nil {
			errs++
			if errs == 1 {
				fmt.Fprintln(stdout, "error:", r.err)
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
	fmt.Fprintf(stdout, "n=%d tokens=%d wall=%.1fs total=%.0f tok/s per-request-median=%.1f tok/s errors=%d\n", *n, total, wall, float64(total)/wall, med, errs)
	if errs != 0 {
		return 1
	}
	return 0
}

func main() {
	os.Exit(runLoad(os.Args[1:], http.DefaultClient, time.Now, os.Stdout, os.Stderr))
}
