package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestProxyKeepsTurnsTogetherAndSeparatesCards(t *testing.T) {
	seen := make(chan string, 3)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen <- r.Header.Get("X-Inky-Card-Id") + ":" + r.URL.Path + ":" + r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: ok\n\n")
	}))
	defer upstream.Close()
	u, _ := url.Parse(upstream.URL)
	for _, id := range []string{"card-a", "card-a", "card-b"} {
		r := httptest.NewRequest("POST", "http://localhost/v1/chat/completions", strings.NewReader("{}"))
		r.Header.Set("X-Inky-Card-Id", "spoofed")
		r.Header.Set("Authorization", "Bearer test-key")
		w := httptest.NewRecorder()
		proxyFor(u, id, "").ServeHTTP(w, r)
		if w.Code != 200 || w.Body.String() != "data: ok\n\n" {
			t.Fatalf("bad response: %d %q", w.Code, w.Body.String())
		}
		if got, want := <-seen, id+":/v1/chat/completions:Bearer test-key"; got != want {
			t.Fatalf("got %q want %q", got, want)
		}
	}
}

func TestValidateRejectsUnsafeConfiguration(t *testing.T) {
	for _, tc := range []struct{ upstream, id, listen string }{
		{"http://user:pass@host", "card", "127.0.0.1:1"},
		{"http://host/v1", "card", "127.0.0.1:1"},
		{"http://host", "bad\nheader", "127.0.0.1:1"},
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

func TestProxyInjectsConfiguredKey(t *testing.T) {
	seen := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen <- r.Header.Get("Authorization") + ":" + r.Header.Get("X-Api-Key")
	}))
	defer upstream.Close()
	u, _ := url.Parse(upstream.URL)
	r := httptest.NewRequest("POST", "http://localhost/v1/messages", strings.NewReader("{}"))
	r.Header.Set("Authorization", "Bearer placeholder")
	r.Header.Set("X-Api-Key", "placeholder")
	proxyFor(u, "card", "upstream-secret").ServeHTTP(httptest.NewRecorder(), r)
	if got := <-seen; got != "Bearer upstream-secret:" {
		t.Fatalf("unexpected auth %q", got)
	}
}
