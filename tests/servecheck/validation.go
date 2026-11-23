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

func (r apiObservation) expect(code int) error {
	if r.code != code {
		return fmt.Errorf("got HTTP %d, want %d", r.code, code)
	}
	if code >= 400 && r.Error.Message == "" {
		return fmt.Errorf("HTTP %d without an error message", code)
	}
	return nil
}

func (g *gate) observe(path, auth string, body []byte) (apiObservation, error) {
	var result apiObservation
	method := http.MethodGet
	if body != nil {
		method = http.MethodPost
	}
	req, err := http.NewRequest(method, g.url+path, bytes.NewReader(body))
	if err != nil {
		return result, err
	}
	if auth != "" {
		req.Header["authorization"] = []string{auth}
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

func (g *gate) validation() {
	g.run("bearer authentication and public health", func() error {
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
		if len(r.Models) == 0 {
			return fmt.Errorf("authorized model list is empty")
		}
		return g.checkHealth("/health")
	})
	g.run("sampling bounds and subsequent health", func() error {
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
	g.run("bearer header casing and trailing-slash health", func() error {
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
