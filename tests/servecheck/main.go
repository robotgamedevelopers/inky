// The server gate starts one server at a time. All requests use an ephemeral loopback port.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
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

const (
	checkTimeout = 2 * time.Minute
	suiteTimeout = 12 * time.Minute
)

type gate struct {
	ctx           context.Context
	url, key, log string
	sabotage      string
	failed        bool
}

func canonicalSabotageFault(raw string) (string, bool) {
	switch raw {
	case "", "all":
		return raw, true
	case "health-status":
		return "health-status", true
	case "ignore-budget", "memory-budget":
		return "ignore-budget", true
	case "skip-thinking-cancel", "disconnect-mid-thinking":
		return "skip-thinking-cancel", true
	case "leak-permit", "concurrent-tool-calls":
		return "leak-permit", true
	default:
		return "", false
	}
}

func (g *gate) faultEnabled(fault string) bool {
	return g.sabotage == "all" || g.sabotage == fault
}

func (g *gate) run(name string, fn func(*gate) error) {
	// Each check owns its context, including any concurrent requests it starts.
	check := *g
	ctx, cancel := context.WithTimeout(g.ctx, checkTimeout)
	defer cancel()
	check.ctx = ctx
	err := ctx.Err()
	if err == nil {
		err = fn(&check)
		if err == nil {
			err = ctx.Err()
		}
	}
	if err != nil {
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

// The server owns the ephemeral listener from bind through shutdown. Read its
// actual address instead of releasing a temporary reservation before launch.
func readyURL(output string) (string, error) {
	for _, line := range strings.Split(output, "\n") {
		rest, ok := strings.CutPrefix(line, "[serve] http://")
		if !ok {
			continue
		}
		address, _, complete := strings.Cut(rest, "/v1:")
		if !complete {
			continue
		}
		host, port, err := net.SplitHostPort(address)
		n, portErr := strconv.Atoi(port)
		if err != nil || host != "127.0.0.1" || portErr != nil || n < 1 || n > 65535 {
			return "", fmt.Errorf("invalid loopback readiness address %q", address)
		}
		return "http://" + address, nil
	}
	return "", nil
}

// Report a bounded server-side diagnostic before the temporary directory is removed.
func serverLogTail(log *os.File) string {
	const limit = 8 << 10
	info, err := log.Stat()
	if err != nil {
		return err.Error()
	}
	tail, err := io.ReadAll(io.NewSectionReader(log, max(0, info.Size()-limit), limit))
	if err != nil {
		return err.Error()
	}
	return string(tail)
}

func (g *gate) server(ctx context.Context, bin, model, dir string, lanes, queue int, checks func(*gate)) error {
	return g.serverWithFault(ctx, bin, model, dir, lanes, queue, "", checks)
}

func (g *gate) serverWithFault(ctx context.Context, bin, model, dir string, lanes, queue int, fault string, checks func(*gate)) (err error) {
	logName := fmt.Sprintf("lanes-%d.log", lanes)
	if fault != "" {
		logName = fmt.Sprintf("fault-%s-lanes-%d.log", fault, lanes)
	}
	g.log = filepath.Join(dir, logName)
	log, err := os.Create(g.log)
	if err != nil {
		return err
	}
	defer log.Close()
	var checksFailed bool
	defer func() {
		if err != nil || checksFailed {
			fmt.Fprintf(os.Stderr, "server diagnostics (%s):\n%s\n", filepath.Base(g.log), serverLogTail(log))
		}
	}()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "serve", "-m", model, "-c", "6144", "-n", "4096", "--lanes", strconv.Itoa(lanes), "--queue", strconv.Itoa(queue), "--key-env", "INKY_SERVE_TEST_KEY", "--effort", "none", "--bind", "127.0.0.1:0")
	cmd.Env = append(os.Environ(), "INKY_SERVE_TEST_KEY="+g.key, "INK_CHUNK=64")
	if fault != "" {
		cmd.Env = append(cmd.Env, "INKY_SERVE_ALLOW_FAULT_INJECTION=1", "INKY_SERVE_FAULT="+fault)
	}
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
		case <-ctx.Done():
			return ctx.Err()
		case <-done:
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return fmt.Errorf("server startup: %v", processErr)
		case <-deadline.C:
			return errors.New("server startup timed out")
		case <-tick.C:
			data, err := os.ReadFile(g.log)
			if err != nil {
				return err
			}
			url, err := readyURL(string(data))
			if err != nil {
				return err
			}
			if url != "" {
				server := *g
				server.ctx, server.url, server.failed = ctx, url, false
				checks(&server)
				checksFailed = server.failed
				g.failed = g.failed || checksFailed
				if ctx.Err() != nil {
					return ctx.Err()
				}
				select {
				case <-done:
					return fmt.Errorf("server exited during checks: %v", processErr)
				default:
					return nil
				}
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
	ctx, cancel := context.WithTimeout(ctx, suiteTimeout)
	defer cancel()
	// Redirects and proxies would take functional requests outside this test server.
	http.DefaultTransport.(*http.Transport).Proxy = nil
	http.DefaultClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	sabotageRaw := os.Getenv("SERVECHECK_SABOTAGE")
	sabotage, ok := canonicalSabotageFault(sabotageRaw)
	if !ok {
		return fmt.Errorf("unknown SERVECHECK_SABOTAGE probe %q", sabotageRaw)
	}
	g := gate{ctx: ctx, key: "local-gate-test", sabotage: sabotage}
	bin, model := os.Args[1], os.Args[2]
	if sabotage != "" {
		// Negative-control mode: plant each requested fault in the real `inky serve` process
		// (gated behind INKY_SERVE_ALLOW_FAULT_INJECTION=1 + INKY_SERVE_FAULT) and run the
		// unmodified production probe against it.
		if g.faultEnabled("health-status") {
			if err = g.serverWithFault(ctx, bin, model, dir, 1, 1, "health-status", func(g *gate) {
				g.run("bearer authentication and public health", (*gate).authAndPublicHealth)
			}); err != nil {
				return err
			}
		}
		if g.faultEnabled("ignore-budget") {
			if err = g.serverWithFault(ctx, bin, model, dir, 1, 1, "ignore-budget", func(g *gate) {
				g.run("health reports Metal memory budget and admission limits", func(g *gate) error {
					return g.healthMemoryBudget(1)
				})
			}); err != nil {
				return err
			}
		}
		if g.faultEnabled("skip-thinking-cancel") {
			if err = g.serverWithFault(ctx, bin, model, dir, 1, 1, "skip-thinking-cancel", func(g *gate) {
				g.run("disconnect in mid-thinking stops generation", (*gate).disconnectMidThinking)
			}); err != nil {
				return err
			}
		}
		if g.faultEnabled("leak-permit") {
			if err = g.serverWithFault(ctx, bin, model, dir, 4, 2, "leak-permit", func(g *gate) {
				g.run("concurrent identical tool calls", (*gate).concurrentToolCalls)
			}); err != nil {
				return err
			}
		}
		if g.failed {
			return errors.New("one or more serve checks failed")
		}
		return nil
	}
	g.run("lane limit checked before model load", func(g *gate) error { return checkLaneRange(g.ctx, bin, dir) })
	// Run the cold reference on a separate server so it cannot seed the warm server's cache.
	window := windowMessages()
	var cold replySample
	if err = g.server(ctx, bin, model, dir, 1, 1, func(g *gate) {
		g.run("cold reference for the retained-window check", func(check *gate) error {
			var err error
			cold, err = check.requestReply(window, 32)
			return err
		})
	}); err != nil {
		return err
	}
	if err = g.server(ctx, bin, model, dir, 1, 1, func(g *gate) {
		g.run("follow-up after window overwrite matches cold generation", func(check *gate) error {
			return check.overwrittenWindow(window, cold)
		})
		g.validation(1)
		g.cancelJobs()
		g.run("disconnect in mid-thinking stops generation", (*gate).disconnectMidThinking)
	}); err != nil {
		return err
	}
	if err = g.server(ctx, bin, model, dir, 2, 1, func(g *gate) {
		g.replies()
		g.run("claude harness /v1/messages multi-turn tool loop and timing", (*gate).claudeHarnessMessagesLoop)
		g.run("opencode harness /v1/chat/completions include_usage and tool loop", (*gate).opencodeHarnessChatLoop)
		g.queueJobs()
	}); err != nil {
		return err
	}
	if err = g.server(ctx, bin, model, dir, 4, 2, func(g *gate) {
		g.run("concurrent identical tool calls", (*gate).concurrentToolCalls)
		g.run("JSON grammar constrained high-temperature tool calls", (*gate).toolGrammarConstrainedSampling)
	}); err != nil {
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
