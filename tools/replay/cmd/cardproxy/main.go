// cardproxy attaches one explicit card ID to all requests from one local agent.
package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"
)

func proxyFor(upstream *url.URL, cardID string) http.Handler {
	p := httputil.NewSingleHostReverseProxy(upstream)
	original := p.Director
	p.Director = func(r *http.Request) {
		original(r)
		r.Host = upstream.Host
		// A client cannot change the identity of this dedicated card proxy.
		r.Header.Set("X-Inky-Card-Id", cardID)
	}
	p.FlushInterval = -1 // preserve streaming responses
	return p
}

func validate(upstream, cardID, listen string) (*url.URL, error) {
	u, err := url.Parse(upstream)
	if err != nil || u == nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return nil, fmt.Errorf("upstream must be an HTTP(S) origin without credentials, query or fragment")
	}
	if u.Path != "" && u.Path != "/" {
		return nil, fmt.Errorf("upstream must be an origin, without a path")
	}
	if cardID == "" || len(cardID) > 128 || strings.TrimSpace(cardID) != cardID {
		return nil, fmt.Errorf("card ID must be 1 to 128 bytes without surrounding whitespace")
	}
	for _, c := range cardID {
		if c < 0x21 || c > 0x7e {
			return nil, fmt.Errorf("card ID must contain printable ASCII without spaces")
		}
	}
	host, _, err := net.SplitHostPort(listen)
	if err != nil || net.ParseIP(host) == nil || !net.ParseIP(host).IsLoopback() {
		return nil, fmt.Errorf("listen must be a loopback IP and port")
	}
	return u, nil
}

func main() {
	upstream := flag.String("upstream", "", "Inky server origin")
	cardID := flag.String("card-id", "", "unique ID for this agent run")
	listen := flag.String("listen", "127.0.0.1:0", "local loopback listen address")
	flag.Parse()
	u, err := validate(*upstream, *cardID, *listen)
	if err != nil {
		log.Fatal(err)
	}
	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("CARD_PROXY_URL=http://%s\n", ln.Addr())
	s := &http.Server{Handler: proxyFor(u, *cardID), ReadHeaderTimeout: 10 * time.Second}
	log.Fatal(s.Serve(ln))
}
