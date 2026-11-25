package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type apiObservation struct {
	code   int
	OK     bool                     `json:"ok"`
	Error  struct{ Message string } `json:"error"`
	Models []struct{ ID string }    `json:"data"`
}

type healthSnapshot struct {
	OK                         bool   `json:"ok"`
	ActiveLanes                *int   `json:"active_lanes"`
	Inflight                   *int   `json:"inflight"`
	RecommendedWorkingSetBytes *int64 `json:"recommended_working_set_bytes"`
	ModelBytes                 *int64 `json:"model_bytes"`
	PerLaneWorstCaseBytes      *int64 `json:"per_lane_worst_case_bytes"`
	MaxAdmittedLanes           *int   `json:"max_admitted_lanes"`
	QueuedRequests             *int   `json:"queued_requests"`
}

func (g *gate) fetchHealth(path string) (int, healthSnapshot, error) {
	var snap healthSnapshot
	req, err := http.NewRequestWithContext(g.ctx, http.MethodGet, g.url+path, nil)
	if err != nil {
		return 0, snap, err
	}
	client := &http.Client{Timeout: 5 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Do(req)
	if err != nil {
		return 0, snap, err
	}
	defer resp.Body.Close()
	err = json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&snap)
	return resp.StatusCode, snap, err
}

func (g *gate) waitHealthDrained(timeout time.Duration) (healthSnapshot, error) {
	deadline := time.Now().Add(timeout)
	var last healthSnapshot
	for {
		code, snap, err := g.fetchHealth("/health")
		if err == nil && code == http.StatusOK {
			last = snap
			if snap.OK &&
				snap.ActiveLanes != nil && *snap.ActiveLanes == 0 &&
				snap.Inflight != nil && *snap.Inflight == 0 &&
				snap.QueuedRequests != nil && *snap.QueuedRequests == 0 {
				return snap, nil
			}
		}
		if g.ctx.Err() != nil {
			return last, g.ctx.Err()
		}
		if time.Now().After(deadline) {
			if err != nil {
				return last, fmt.Errorf("/health drain check failed: %w", err)
			}
			return last, fmt.Errorf("/health did not drain within %v: active_lanes=%v inflight=%v queued=%v",
				timeout, ptrInt(last.ActiveLanes), ptrInt(last.Inflight), ptrInt(last.QueuedRequests))
		}
		time.Sleep(25 * time.Millisecond)
	}
}

func ptrInt(p *int) string {
	if p == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%d", *p)
}

func verifyHealthBudget(snap healthSnapshot, expectedMaxLanes int) error {
	if !snap.OK {
		return fmt.Errorf("/health ok=false")
	}
	if snap.RecommendedWorkingSetBytes == nil || *snap.RecommendedWorkingSetBytes <= 0 {
		return fmt.Errorf("missing or non-positive recommended_working_set_bytes")
	}
	if snap.ModelBytes == nil || *snap.ModelBytes <= 0 {
		return fmt.Errorf("missing or non-positive model_bytes")
	}
	if *snap.ModelBytes >= *snap.RecommendedWorkingSetBytes {
		return fmt.Errorf("model_bytes (%d) >= recommended_working_set_bytes (%d)", *snap.ModelBytes, *snap.RecommendedWorkingSetBytes)
	}
	if snap.PerLaneWorstCaseBytes == nil || *snap.PerLaneWorstCaseBytes <= 0 {
		return fmt.Errorf("missing or non-positive per_lane_worst_case_bytes")
	}
	if snap.MaxAdmittedLanes == nil || *snap.MaxAdmittedLanes < 1 {
		return fmt.Errorf("missing or invalid max_admitted_lanes")
	}
	if expectedMaxLanes > 0 && *snap.MaxAdmittedLanes != expectedMaxLanes {
		return fmt.Errorf("max_admitted_lanes=%d, want %d", *snap.MaxAdmittedLanes, expectedMaxLanes)
	}
	if snap.ActiveLanes == nil || *snap.ActiveLanes != 0 {
		return fmt.Errorf("expected idle active_lanes=0, got %s", ptrInt(snap.ActiveLanes))
	}
	if snap.Inflight == nil || *snap.Inflight != 0 {
		return fmt.Errorf("expected idle inflight=0, got %s", ptrInt(snap.Inflight))
	}
	if snap.QueuedRequests == nil || *snap.QueuedRequests != 0 {
		return fmt.Errorf("expected idle queued_requests=0, got %s", ptrInt(snap.QueuedRequests))
	}
	return nil
}

func (r apiObservation) expect(code int) error {
	if r.code != code {
		return fmt.Errorf("got HTTP %d, want %d", r.code, code)
	}
	if code >= 400 && r.Error.Message == "" {
		return fmt.Errorf("HTTP %d without an error message", code)
	}
	return nil
}

func verifyModelsList(models []struct{ ID string }) error {
	seen := make(map[string]bool, len(models))
	for _, m := range models {
		seen[m.ID] = true
	}
	for _, want := range []string{"inkling-2-mini", "inky/inkling-2-mini"} {
		if !seen[want] {
			return fmt.Errorf("/v1/models missing required model id %q (got %+v)", want, models)
		}
	}
	return nil
}

func (g *gate) observeHeaders(path string, headers map[string]string, body []byte) (apiObservation, error) {
	var result apiObservation
	method := http.MethodGet
	if body != nil {
		method = http.MethodPost
	}
	req, err := http.NewRequestWithContext(g.ctx, method, g.url+path, bytes.NewReader(body))
	if err != nil {
		return result, err
	}
	for k, v := range headers {
		req.Header[k] = []string{v}
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	client := &http.Client{Timeout: 5 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Do(req)
	if err != nil {
		return result, err
	}
	defer resp.Body.Close()
	result.code = resp.StatusCode
	err = json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&result)
	return result, err
}

func (g *gate) observe(path, auth string, body []byte) (apiObservation, error) {
	headers := map[string]string{}
	if auth != "" {
		headers["authorization"] = auth
	}
	return g.observeHeaders(path, headers, body)
}

func (g *gate) checkHealth(path string) error {
	r, err := g.observe(path, "", nil)
	if err != nil {
		return err
	}
	if err := r.expect(http.StatusOK); err != nil {
		return err
	}
	if !r.OK {
		return fmt.Errorf("%s did not report ok", path)
	}
	return nil
}

func (g *gate) authAndPublicHealth() error {
	for _, auth := range []string{"", "Bearer " + g.key + "-wrong"} {
		r, err := g.observe("/v1/models", auth, nil)
		if err != nil {
			return err
		}
		if err := r.expect(401); err != nil {
			return err
		}
		if r.expect(200) == nil {
			return fmt.Errorf("accepted incorrect anonymous success observation")
		}
	}
	r, err := g.observe("/v1/models", "Bearer "+g.key, nil)
	if err != nil {
		return err
	}
	if err := r.expect(200); err != nil {
		return err
	}
	if err := verifyModelsList(r.Models); err != nil {
		return err
	}
	// x-api-key authentication works identically to Bearer and rejects invalid keys with 401.
	badKey, err := g.observeHeaders("/v1/models", map[string]string{"x-api-key": g.key + "-wrong"}, nil)
	if err != nil {
		return err
	}
	if err := badKey.expect(401); err != nil {
		return fmt.Errorf("bad x-api-key: %w", err)
	}
	goodKey, err := g.observeHeaders("/v1/models", map[string]string{"x-api-key": g.key}, nil)
	if err != nil {
		return err
	}
	if err := goodKey.expect(200); err != nil {
		return fmt.Errorf("good x-api-key: %w", err)
	}
	if err := verifyModelsList(goodKey.Models); err != nil {
		return err
	}
	return g.checkHealth("/health")
}

func (g *gate) healthMemoryBudget(expectedLanes int) error {
	code, snap, err := g.fetchHealth("/health")
	if err != nil {
		return err
	}
	if code != http.StatusOK {
		return fmt.Errorf("got HTTP %d, want 200", code)
	}
	// Self-check negative witness: zero working set must be rejected.
	zero := int64(0)
	badSnap := snap
	badSnap.RecommendedWorkingSetBytes = &zero
	if verifyHealthBudget(badSnap, expectedLanes) == nil {
		return fmt.Errorf("health budget assertion accepted zero working set")
	}
	return verifyHealthBudget(snap, expectedLanes)
}

func (g *gate) validation(expectedLanes int) {
	g.run("bearer authentication and public health", (*gate).authAndPublicHealth)
	g.run("health reports Metal memory budget and admission limits", func(g *gate) error {
		return g.healthMemoryBudget(expectedLanes)
	})
	g.run("sampling bounds and subsequent health", func(g *gate) error {
		for _, tc := range []struct{ name, field string }{{"max_tokens", `"max_tokens":18446744073709551615`}, {"temperature", `"temperature":5,"max_tokens":1`}, {"top_p", `"top_p":0,"max_tokens":1`}} {
			body := []byte(`{"messages":[{"role":"user","content":"Hello."}],` + tc.field + "}")
			r, err := g.observe("/v1/chat/completions", "Bearer "+g.key, body)
			if err != nil {
				return err
			}
			if err := r.expect(400); err != nil {
				return fmt.Errorf("%s: %w", tc.name, err)
			}
			if !strings.Contains(r.Error.Message, tc.name) {
				return fmt.Errorf("%s: unexpected error %q", tc.name, r.Error.Message)
			}
			if r.expect(200) == nil {
				return fmt.Errorf("accepted incorrect sampling success observation")
			}
		}
		return g.checkHealth("/health")
	})
	g.run("bearer header casing and trailing-slash health", func(g *gate) error {
		r, err := g.observe("/v1/models", "BEARER "+g.key, nil)
		if err != nil {
			return err
		}
		if err := r.expect(200); err != nil {
			return err
		}
		if len(r.Models) == 0 {
			return fmt.Errorf("authorized model list is empty")
		}
		if r.expect(401) == nil {
			return fmt.Errorf("accepted incorrect rejection observation")
		}
		return g.checkHealth("/health/")
	})
}
