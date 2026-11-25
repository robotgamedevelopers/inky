// xsim: does choosing which lanes decode together, by predicted expert overlap, cut the bytes read per step?
// Input: inkling's ref/routes dumps (G lines: token, 48 layer passes of top-8 experts).
package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

const L, E = 48, 128
const sharedGB, expertGB = 1.26, (2.986 + 1.66) / 1000 // from inkling's tools/spec_analysis.py

type tok [L]uint128

type uint128 struct{ lo, hi uint64 }

func (a *uint128) set(e int) {
	if e < 64 {
		a.lo |= 1 << e
	} else {
		a.hi |= 1 << (e - 64)
	}
}
func or(a, b uint128) uint128 { return uint128{a.lo | b.lo, a.hi | b.hi} }
func pop(a uint128) int       { return popc(a.lo) + popc(a.hi) }
func popc(x uint64) int {
	n := 0
	for ; x != 0; x &= x - 1 {
		n++
	}
	return n
}

func loadReader(r io.Reader) ([]tok, error) {
	var out []tok
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 4<<20)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		parts := strings.Fields(sc.Text())
		if len(parts) == 0 || parts[0] == "P" {
			continue
		}
		if parts[0] != "G" {
			return nil, fmt.Errorf("line %d: unexpected record tag %q", lineNo, parts[0])
		}
		if len(parts) < 3 {
			return nil, fmt.Errorf("line %d: malformed G record (need token and %d layer passes)", lineNo, L)
		}
		if _, err := strconv.Atoi(parts[1]); err != nil {
			return nil, fmt.Errorf("line %d: invalid token id %q: %w", lineNo, parts[1], err)
		}
		passes := strings.Split(parts[2], ";")
		if len(passes) != L {
			return nil, fmt.Errorf("line %d: got %d layer passes, want %d", lineNo, len(passes), L)
		}
		var t tok
		for l, s := range passes {
			if s == "" {
				return nil, fmt.Errorf("line %d: empty expert list at pass %d", lineNo, l)
			}
			for _, e := range strings.Split(s, ",") {
				v, err := strconv.Atoi(e)
				if err != nil || v < 0 || v >= E {
					return nil, fmt.Errorf("line %d: invalid expert id %q at pass %d (want 0..%d)", lineNo, e, l, E-1)
				}
				t[l].set(v)
			}
		}
		out = append(out, t)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(out) < 2 {
		return nil, fmt.Errorf("need at least 2 generated (G) tokens, got %d", len(out))
	}
	return out, nil
}

func load(p string) ([]tok, error) {
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	toks, err := loadReader(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", p, err)
	}
	return toks, nil
}

type lane struct{ f, i int } // file, position (i>=1 so i-1 predicts)

func distinct(files [][]tok, ls []lane, prev bool) [L]uint128 {
	var u [L]uint128
	for _, x := range ls {
		i := x.i
		if prev {
			i--
		}
		for l := 0; l < L; l++ {
			u[l] = or(u[l], files[x.f][i][l])
		}
	}
	return u
}
func cost(files [][]tok, ls []lane) float64 {
	u := distinct(files, ls, false)
	n := 0
	for l := 0; l < L; l++ {
		n += pop(u[l])
	}
	return float64(n) / L
}

func randLane(files [][]tok, r *rand.Rand, f int) lane {
	if f < 0 {
		f = r.Intn(len(files))
	}
	return lane{f, 1 + r.Intn(len(files[f])-1)}
}

// greedy: from pool, seed with pool[0] (the oldest waiter), then add the lane whose predicted (previous-token) experts add fewest new ones
func greedy(files [][]tok, pool []lane, B int, oracle bool) []lane {
	if len(pool) == 0 || B <= 0 {
		return nil
	}
	if B > len(pool) {
		B = len(pool)
	}
	used := make([]bool, len(pool))
	batch := []lane{pool[0]}
	used[0] = true
	u := distinct(files, batch, !oracle)
	for len(batch) < B {
		best, bi := math.MaxInt, -1
		for j, c := range pool {
			if used[j] {
				continue
			}
			d := distinct(files, []lane{c}, !oracle)
			n := 0
			for l := 0; l < L; l++ {
				n += pop(or(u[l], d[l])) - pop(u[l])
			}
			if n < best {
				best, bi = n, j
			}
		}
		if bi < 0 {
			break
		}
		used[bi] = true
		batch = append(batch, pool[bi])
		d := distinct(files, []lane{pool[bi]}, !oracle)
		for l := 0; l < L; l++ {
			u[l] = or(u[l], d[l])
		}
	}
	return batch
}

func gb(x float64) float64 { return sharedGB + L*x*expertGB }

func runXsim(paths []string, trials int, stdout, stderr io.Writer) int {
	if len(paths) == 0 {
		fmt.Fprintln(stderr, "usage: xsim <routes-file>...")
		return 2
	}
	if trials < 1 {
		fmt.Fprintln(stderr, "trials must be positive")
		return 2
	}
	var files [][]tok
	for _, p := range paths {
		toks, err := load(p)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		files = append(files, toks)
	}
	r := rand.New(rand.NewSource(1))
	fmt.Fprintln(stdout, "experts per layer pass touched by one decode step of B lanes (mean of", trials, "trials); weight GB per step in brackets; K/V reads not included")
	fmt.Fprintf(stdout, "%4s %10s %16s %16s %16s %16s %16s\n", "B", "uniform", "random mix", "same task", "greedy pool 4B", "oracle pool 4B", "greedy pool 16B")
	for _, B := range []int{1, 4, 8, 16, 32, 64, 128} {
		var rm, st, g4, o4, g16 float64
		for t := 0; t < trials; t++ {
			mk := func(n, f int) []lane {
				ls := make([]lane, n)
				for i := range ls {
					ls[i] = randLane(files, r, f)
				}
				return ls
			}
			rm += cost(files, mk(B, -1))
			st += cost(files, mk(B, r.Intn(len(files))))
			p4 := mk(4*B, -1)
			g4 += cost(files, greedy(files, p4, B, false))
			o4 += cost(files, greedy(files, p4, B, true))
			if B <= 32 {
				g16 += cost(files, greedy(files, mk(16*B, -1), B, false))
			}
		}
		n := float64(trials)
		uni := E * (1 - math.Pow(1-8.0/E, float64(B)))
		cell := func(x float64) string { return fmt.Sprintf("%5.1f [%5.1f]", x, gb(x)) }
		g16s := "-"
		if B <= 32 {
			g16s = cell(g16 / n)
		}
		fmt.Fprintf(stdout, "%4d %10s %16s %16s %16s %16s %16s\n", B, cell(uni), cell(rm/n), cell(st/n), cell(g4/n), cell(o4/n), g16s)
	}
	return 0
}

func main() {
	os.Exit(runXsim(os.Args[1:], 300, os.Stdout, os.Stderr))
}
