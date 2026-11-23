// The server gate starts one server at a time. All requests use an ephemeral loopback port.
package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type gate struct {
	url, key, log string
	sabotage      string
	failed        bool
}

func (g *gate) run(name string, fn func() error) {
	if err := fn(); err != nil {
		fmt.Printf("FAIL %s: %v\n", name, err)
		g.failed = true
		return
	}
	fmt.Printf("PASS %s\n", name)
}

func laneRange(exit int, output string) bool {
	return exit == 2 && strings.Contains(output, "--lanes must be between 1 and 128")
}

func checkLaneRange(ctx context.Context, bin, dir string) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, bin, "serve", "-m", filepath.Join(dir, "absent.gguf"), "--lanes", "200").CombinedOutput()
	var status *exec.ExitError
	if !errors.As(err, &status) || !laneRange(status.ExitCode(), string(out)) {
		return fmt.Errorf("expected usage exit and lane range; got %v, %s", err, out)
	}
	if laneRange(0, string(out)) || laneRange(2, "failed to load model") {
		return errors.New("lane range assertion accepted wrong observation")
	}
	return nil
}

func (g *gate) server(ctx context.Context, bin, model, dir string, lanes int, checks func()) error {
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		return err
	}
	address := listener.Addr().String()
	listener.Close()
	g.url = "http://" + address
	g.log = filepath.Join(dir, fmt.Sprintf("lanes-%d.log", lanes))
	log, err := os.Create(g.log)
	if err != nil {
		return err
	}
	defer log.Close()
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "serve", "-m", model, "-c", "6144", "-n", "4096", "--lanes", strconv.Itoa(lanes), "--queue", "1", "--key-env", "INKY_SERVE_TEST_KEY", "--effort", "none", "--bind", address)
	cmd.Env = append(os.Environ(), "INKY_SERVE_TEST_KEY="+g.key, "INK_CHUNK=64")
	cmd.Stdout, cmd.Stderr = log, log
	if err = cmd.Start(); err != nil {
		return err
	}
	done := make(chan struct{})
	var processErr error
	go func() { processErr = cmd.Wait(); close(done) }()
	defer func() { cancel(); <-done }()
	deadline := time.NewTimer(time.Minute)
	defer deadline.Stop()
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-done:
			data, _ := os.ReadFile(g.log)
			return fmt.Errorf("server startup: %v\n%s", processErr, data)
		case <-deadline.C:
			return errors.New("server startup timed out")
		case <-tick.C:
			data, err := os.ReadFile(g.log)
			if err != nil {
				return err
			}
			if strings.Contains(string(data), "[serve] "+g.url+"/v1:") {
				checks()
				if ctx.Err() != nil {
					return ctx.Err()
				}
				return nil
			}
		}
	}
}

func execute() error {
	if len(os.Args) != 3 {
		return errors.New("expected binary and model paths")
	}
	dir, err := os.MkdirTemp("", "inky-serve-gate-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	// Redirects and proxies would take functional requests outside this test server.
	http.DefaultTransport.(*http.Transport).Proxy = nil
	http.DefaultClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	sabotage := os.Getenv("SERVECHECK_SABOTAGE")
	if sabotage != "" && sabotage != "health-status" {
		return fmt.Errorf("unknown SERVECHECK_SABOTAGE probe %q", sabotage)
	}
	g := gate{key: "local-gate-test", sabotage: sabotage}
	bin, model := os.Args[1], os.Args[2]
	g.run("lane limit checked before model load", func() error { return checkLaneRange(ctx, bin, dir) })
	if err = g.server(ctx, bin, model, dir, 1, func() { g.validation(); g.cancelJobs() }); err != nil {
		return err
	}
	if err = g.server(ctx, bin, model, dir, 2, func() { g.replies(); g.queueJobs() }); err != nil {
		return err
	}
	if g.failed {
		return errors.New("one or more serve checks failed")
	}
	return nil
}

func main() {
	if err := execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
