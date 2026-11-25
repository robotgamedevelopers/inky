package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptrace"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const countPrompt = "Write the integers 1 through 10000 in order, separated by commas. Do not summarize."
const laneStartLimit = 3 * time.Second

func (g *gate) jobRequest(ctx context.Context, prompt string, limit int, stream bool) (*http.Response, error) {
	payload := map[string]any{"messages": []map[string]string{{"role": "user", "content": prompt}}, "max_tokens": limit, "stream": stream, "seed": 1729, "temperature": 0.8}
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", g.url+"/v1/chat/completions", strings.NewReader(string(data)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+g.key)
	return http.DefaultClient.Do(req)
}

type jobStream struct {
	body  io.ReadCloser
	lines *bufio.Scanner
}

func (g *gate) startStream(ctx context.Context, prompt string, limit int) (*jobStream, error) {
	res, err := g.jobRequest(ctx, prompt, limit, true)
	if err != nil {
		return nil, err
	}
	if res.StatusCode != 200 || !strings.HasPrefix(res.Header.Get("Content-Type"), "text/event-stream") {
		res.Body.Close()
		return nil, fmt.Errorf("expected accepted stream, got HTTP %s", res.Status)
	}
	return &jobStream{res.Body, bufio.NewScanner(res.Body)}, nil
}

type jobEvent struct {
	Choices []struct {
		Delta struct {
			Content   string
			Reasoning string `json:"reasoning_content"`
		}
		Finish string `json:"finish_reason"`
	}
	Usage struct {
		Tokens int `json:"completion_tokens"`
	}
}

func (s *jobStream) read(firstDelta bool) error {
	tokens := 0
	for s.lines.Scan() {
		line := s.lines.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			return fmt.Errorf("unexpected SSE line: %q", line)
		}
		if data == "[DONE]" {
			if !firstDelta && tokens > 0 {
				return nil
			}
			return fmt.Errorf("stream ended without expected output")
		}
		var e jobEvent
		if err := json.Unmarshal([]byte(data), &e); err != nil {
			return err
		}
		if len(e.Choices) != 1 {
			return fmt.Errorf("missing completion choice: %s", data)
		}
		c := e.Choices[0]
		if firstDelta && (c.Delta.Content != "" || c.Delta.Reasoning != "") {
			return nil
		}
		if c.Finish == "length" || c.Finish == "stop" {
			tokens = e.Usage.Tokens
		}
	}
	if err := s.lines.Err(); err != nil {
		return err
	}
	return io.ErrUnexpectedEOF
}

// Take the log offset before the sole disconnect in this trial. Cancellation lines
// have no request ID. The isolated server and a fresh interval establish attribution.
func (g *gate) logOffset() (int64, error) {
	s, e := os.Stat(g.log)
	if e != nil {
		return 0, e
	}
	return s.Size(), nil
}

var decodeFreed = regexp.MustCompile(`\[serve\] client gone after ([0-9]+) generated tokens; lane freed`)
var prefillFreed = regexp.MustCompile(`\[serve\] lane [0-9]+: client gone during prefill \(([0-9]+) of ([0-9]+) prompt tokens\); lane freed`)

func (g *gate) confirmFreed(offset int64, prefill bool, cap int) error {
	f, e := os.Open(g.log)
	if e != nil {
		return e
	}
	defer f.Close()
	if _, e = f.Seek(offset, io.SeekStart); e != nil {
		return e
	}
	b, e := io.ReadAll(f)
	if e != nil {
		return e
	}
	wrong := fmt.Sprintf("[serve] client gone after %d generated tokens; lane freed", cap)
	if prefill {
		wrong = "[serve] lane 0: client gone during prefill (100 of 100 prompt tokens); lane freed"
	}
	if checkFreed([]byte(wrong), prefill, cap) == nil {
		return fmt.Errorf("cancellation assertion accepted exhausted budget")
	}
	return checkFreed(b, prefill, cap)
}

func checkFreed(b []byte, prefill bool, cap int) error {
	matches := decodeFreed.FindSubmatch(b)
	if prefill {
		matches = prefillFreed.FindSubmatch(b)
	}
	if matches == nil {
		return fmt.Errorf("missing cancellation receipt in new log: %s", b)
	}
	n, e := strconv.Atoi(string(matches[1]))
	if e != nil {
		return e
	}
	if prefill {
		total, e := strconv.Atoi(string(matches[2]))
		if e != nil {
			return e
		}
		if n <= 0 || n >= total {
			return fmt.Errorf("prefill did not stop early: %d/%d", n, total)
		}
	} else if n <= 0 || n >= cap/4 {
		return fmt.Errorf("late decode cancellation: %d/%d tokens", n, cap)
	}
	return nil
}

func (g *gate) canary(ctx context.Context, start time.Time) error {
	s, e := g.startStream(ctx, "What is two plus two?", 12)
	if e != nil {
		return e
	}
	defer s.body.Close()
	if e = s.read(true); e != nil {
		return e
	}
	if elapsed := time.Since(start); elapsed > laneStartLimit {
		return fmt.Errorf("new request waited %v for first token (limit %v)", elapsed, laneStartLimit)
	}
	return s.read(false)
}

func (g *gate) cancelJobs() {
	g.run("stream disconnect frees a lane", func(g *gate) error {
		ctx, cancel := context.WithTimeout(g.ctx, 20*time.Second)
		defer cancel()
		offset, e := g.logOffset()
		if e != nil {
			return e
		}
		s, e := g.startStream(ctx, countPrompt, 256)
		if e != nil {
			return e
		}
		defer s.body.Close()
		if e = s.read(true); e != nil {
			return e
		}
		start := time.Now()
		s.body.Close()
		if e = g.canary(ctx, start); e != nil {
			return e
		}
		return g.confirmFreed(offset, false, 256)
	})
	g.run("nonstream disconnect frees a lane during decode and prefill", func(g *gate) error {
		for _, prefill := range []bool{false, true} {
			if e := g.cancelBuffered(prefill); e != nil {
				return fmt.Errorf("prefill=%v: %w", prefill, e)
			}
		}
		return nil
	})
}

func (g *gate) cancelBuffered(prefill bool) error {
	ctx, cancel := context.WithTimeout(g.ctx, 25*time.Second)
	defer cancel()
	prompt, limit := countPrompt, 4000
	if prefill {
		prompt = strings.Repeat(" apple", 2600)
		limit = 12
	}
	offset, e := g.logOffset()
	if e != nil {
		return e
	}
	wrote := make(chan error, 1)
	traced := httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{WroteRequest: func(i httptrace.WroteRequestInfo) {
		select {
		case wrote <- i.Err:
		default:
		}
	}})
	requestCtx, disconnect := context.WithCancel(traced)
	defer disconnect()
	ended := make(chan error, 1)
	go func() {
		r, e := g.jobRequest(requestCtx, prompt, limit, false)
		if r != nil {
			r.Body.Close()
			if e == nil {
				e = fmt.Errorf("received HTTP %d before disconnect", r.StatusCode)
			}
		}
		ended <- e
	}()
	select {
	case e = <-wrote:
		if e != nil {
			return e
		}
	case e = <-ended:
		return e
	case <-ctx.Done():
		return ctx.Err()
	}
	delay := 2 * time.Second
	if prefill {
		// Give the server time to parse the prompt and start its first chunk.
		// The scheduler receipt checks the actual phase at cancellation.
		delay = 500 * time.Millisecond
	}
	select {
	case e = <-ended:
		return e
	case <-time.After(delay):
	}
	start := time.Now()
	disconnect()
	select {
	case <-ended:
	case <-ctx.Done():
		return ctx.Err()
	}
	if e = g.canary(ctx, start); e != nil {
		return e
	}
	return g.confirmFreed(offset, prefill, limit)
}

type queueResult struct {
	status int
	retry  string
	body   string
}

func (r queueResult) check() error {
	n, e := strconv.Atoi(r.retry)
	if r.status != 429 || e != nil || n <= 0 {
		return fmt.Errorf("got HTTP %d, Retry-After %q", r.status, r.retry)
	}
	if r.body != "" {
		var parsed struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(r.body), &parsed); err != nil || parsed.Error.Message == "" {
			return fmt.Errorf("429 response lacks OpenAI JSON error message: %q", r.body)
		}
	}
	return nil
}

func (g *gate) queueJobs() {
	var fifth queueResult
	var firstPass bool
	g.run("two lanes and one queued request", func(g *gate) error {
		ctx, cancel := context.WithTimeout(g.ctx, 60*time.Second)
		defer cancel()
		var streams []*jobStream
		for _, limit := range []int{256, 256} {
			s, e := g.startStream(ctx, countPrompt, limit)
			if e != nil {
				return e
			}
			defer s.body.Close()
			streams = append(streams, s)
			if e = s.read(true); e != nil {
				return e
			}
		}
		// Third request enters the bounded admission wait queue (blocks until a lane frees).
		type queuedOutcome struct {
			stream *jobStream
			err    error
		}
		queuedCh := make(chan queuedOutcome, 1)
		go func() {
			s, e := g.startStream(ctx, countPrompt, 12)
			queuedCh <- queuedOutcome{s, e}
		}()
		waitDeadline := time.Now().Add(5 * time.Second)
		observedQueued := false
		for ctx.Err() == nil && time.Now().Before(waitDeadline) {
			code, snap, err := g.fetchHealth("/health")
			if err == nil && code == http.StatusOK &&
				snap.ActiveLanes != nil && *snap.ActiveLanes == 2 &&
				snap.QueuedRequests != nil && *snap.QueuedRequests == 1 &&
				snap.Inflight != nil && *snap.Inflight == 3 {
				observedQueued = true
				break
			}
			time.Sleep(15 * time.Millisecond)
		}
		if !observedQueued {
			return fmt.Errorf("third request did not appear in /health as queued_requests=1 active_lanes=2 inflight=3")
		}
		var overflow [2]queueResult
		for i := range overflow {
			r, e := g.jobRequest(ctx, countPrompt, 4, true)
			if e != nil {
				return e
			}
			bodyBytes, _ := io.ReadAll(io.LimitReader(r.Body, 4096))
			overflow[i] = queueResult{r.StatusCode, r.Header.Get("Retry-After"), string(bodyBytes)}
			r.Body.Close()
		}
		fifth = overflow[1]
		if (queueResult{200, "5", ""}).check() == nil {
			return fmt.Errorf("queue assertion accepted fourth request")
		}
		if e := overflow[0].check(); e != nil {
			return e
		}
		for i, s := range streams {
			if e := s.read(false); e != nil {
				return fmt.Errorf("admitted request %d: %w", i+1, e)
			}
		}
		q3 := <-queuedCh
		if q3.err != nil {
			return fmt.Errorf("queued third request failed to admit: %w", q3.err)
		}
		defer q3.stream.body.Close()
		if e := q3.stream.read(false); e != nil {
			return fmt.Errorf("admitted request 3: %w", e)
		}
		if _, e := g.waitHealthDrained(5 * time.Second); e != nil {
			return e
		}
		firstPass = true
		return nil
	})
	g.run("fifth request rejected and capacity restored", func(g *gate) error {
		if !firstPass {
			return fmt.Errorf("first three requests did not complete")
		}
		if (queueResult{429, "", ""}).check() == nil {
			return fmt.Errorf("queue assertion accepted missing retry header")
		}
		if e := fifth.check(); e != nil {
			return e
		}
		ctx, cancel := context.WithTimeout(g.ctx, 20*time.Second)
		defer cancel()
		errs := make(chan error, 3)
		for range 3 {
			go func() {
				s, err := g.startStream(ctx, countPrompt, 32)
				if err != nil {
					errs <- err
					return
				}
				defer s.body.Close()
				errs <- s.read(false)
			}()
		}
		for range 3 {
			if err := <-errs; err != nil {
				return err
			}
		}
		_, err := g.waitHealthDrained(5 * time.Second)
		return err
	})
}
