package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

func TestProxyKeepsTurnsTogetherAndSeparatesCards(t *testing.T) {
	seen := make(chan string, 3)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen <- r.Header.Get("X-Inky-Card-Id") + ":" + r.URL.RequestURI() + ":" + r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: ok\n\n")
	}))
	defer upstream.Close()
	u, _ := url.Parse(upstream.URL)
	proxies := map[string]http.Handler{"card-a": proxyFor(u, "card-a"), "card-b": proxyFor(u, "card-b")}
	for _, id := range []string{"card-a", "card-a", "card-b"} {
		const path = "/v1/chat/completions?stream=true"
		r := httptest.NewRequest("POST", "http://localhost"+path, strings.NewReader("{}"))
		r.Header.Set("X-Inky-Card-Id", "spoofed")
		r.Header.Set("Authorization", "Bearer test-key")
		w := httptest.NewRecorder()
		proxies[id].ServeHTTP(w, r)
		if w.Code != 200 || w.Body.String() != "data: ok\n\n" {
			t.Fatalf("bad response: %d %q", w.Code, w.Body.String())
		}
		if got, want := <-seen, id+":"+path+":Bearer test-key"; got != want {
			t.Fatalf("got %q want %q", got, want)
		}
	}
}

func TestProxyStreamsFirstFrameBeforeUpstreamFinishes(t *testing.T) {
	const first = "data: first\n\n"
	const last = "data: [DONE]\n\n"
	receivedFirst := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, first); err != nil {
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Errorf("flush first frame: %v", err)
			return
		}
		select {
		case <-receivedFirst:
			io.WriteString(w, last)
		case <-r.Context().Done():
		}
	}))
	defer upstream.Close()
	u, _ := url.Parse(upstream.URL)
	gateway := httptest.NewServer(proxyFor(u, "card-stream"))
	defer gateway.Close()

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, gateway.URL+"/v1/chat/completions", strings.NewReader("{}"))
	if err != nil {
		t.Fatal(err)
	}
	res, err := gateway.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	frame := make([]byte, len(first))
	if _, err := io.ReadFull(res.Body, frame); err != nil || string(frame) != first {
		t.Fatalf("first frame %q, error %v", frame, err)
	}
	close(receivedFirst)
	tail, err := io.ReadAll(res.Body)
	if err != nil || string(tail) != last {
		t.Fatalf("terminal frame %q, error %v", tail, err)
	}
}

func TestValidateRejectsUnsafeConfiguration(t *testing.T) {
	for _, tc := range []struct{ upstream, id, listen string }{
		{"http://user:pass@host", "card", "127.0.0.1:1"},
		{"http://host/v1", "card", "127.0.0.1:1"},
		{"http://host", "bad\nheader", "127.0.0.1:1"},
		{"http://host", "", "127.0.0.1:1"},
		{"http://host", "card one", "127.0.0.1:1"},
		{"http://host", "card", "0.0.0.0:1"},
	} {
		if _, err := validate(tc.upstream, tc.id, tc.listen); err == nil {
			t.Fatalf("accepted %+v", tc)
		}
	}
	if _, err := validate("http://host:18096", "card-1", "127.0.0.1:0"); err != nil {
		t.Fatal(err)
	}
}

func TestAnonymousCallerDoesNotAcquireAuthorization(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer expected" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()
	u, _ := url.Parse(upstream.URL)
	gateway := proxyFor(u, "card")
	for _, tc := range []struct {
		auth string
		want int
	}{{"", http.StatusUnauthorized}, {"Bearer expected", http.StatusNoContent}} {
		r := httptest.NewRequest(http.MethodPost, "http://localhost/v1/chat/completions", nil)
		if tc.auth != "" {
			r.Header.Set("Authorization", tc.auth)
		}
		w := httptest.NewRecorder()
		gateway.ServeHTTP(w, r)
		if w.Code != tc.want {
			t.Fatalf("authorization present=%t: status %d, want %d", tc.auth != "", w.Code, tc.want)
		}
	}
}
