// xsim: does choosing which lanes decode together, by predicted expert overlap, cut the bytes read per step?
// Input: inkling's ref/routes dumps (G lines: token, 48 layer passes of top-8 experts).
package main

import (
	"bufio"
	"fmt"
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

func load(p string) []tok {
	f, _ := os.Open(p)
	defer f.Close()
	var out []tok
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		parts := strings.Fields(sc.Text())
		if parts[0] != "G" {
			continue
		}
		var t tok
		for l, s := range strings.Split(parts[2], ";") {
			for _, e := range strings.Split(s, ",") {
				v, _ := strconv.Atoi(e)
				t[l].set(v)
			}
		}
		out = append(out, t)
	}
	return out
}

type lane struct{ f, i int } // file, position (i>=1 so i-1 predicts)

var files [][]tok

func distinct(ls []lane, prev bool) [L]uint128 {
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
func cost(ls []lane) float64 {
	u := distinct(ls, false)
	n := 0
	for l := 0; l < L; l++ {
		n += pop(u[l])
	}
	return float64(n) / L
}

func randLane(r *rand.Rand, f int) lane {
	if f < 0 {
		f = r.Intn(len(files))
	}
	return lane{f, 1 + r.Intn(len(files[f])-1)}
}

// greedy: from pool, seed with pool[0] (the oldest waiter), then add the lane whose predicted (previous-token) experts add fewest new ones
func greedy(pool []lane, B int, oracle bool) []lane {
	used := make([]bool, len(pool))
	batch := []lane{pool[0]}
	used[0] = true
	u := distinct(batch, !oracle)
	for len(batch) < B {
		best, bi := math.MaxInt, -1
		for j, c := range pool {
			if used[j] {
				continue
			}
			d := distinct([]lane{c}, !oracle)
			n := 0
			for l := 0; l < L; l++ {
				n += pop(or(u[l], d[l])) - pop(u[l])
			}
			if n < best {
				best, bi = n, j
			}
		}
		used[bi] = true
		batch = append(batch, pool[bi])
		d := distinct([]lane{pool[bi]}, !oracle)
		for l := 0; l < L; l++ {
			u[l] = or(u[l], d[l])
		}
	}
	return batch
}

func gb(x float64) float64 { return sharedGB + L*x*expertGB }

func main() {
	for _, p := range os.Args[1:] {
		files = append(files, load(p))
	}
	r := rand.New(rand.NewSource(1))
	const trials = 300
	fmt.Println("experts per layer pass touched by one decode step of B lanes (mean of", trials, "trials); weight GB per step in brackets; K/V reads not included")
	fmt.Printf("%4s %10s %16s %16s %16s %16s %16s\n", "B", "uniform", "random mix", "same task", "greedy pool 4B", "oracle pool 4B", "greedy pool 16B")
	for _, B := range []int{1, 4, 8, 16, 32, 64, 128} {
		var rm, st, g4, o4, g16 float64
		for t := 0; t < trials; t++ {
			mk := func(n, f int) []lane {
				ls := make([]lane, n)
				for i := range ls {
					ls[i] = randLane(r, f)
				}
				return ls
			}
			rm += cost(mk(B, -1))
			st += cost(mk(B, r.Intn(len(files))))
			p4 := mk(4*B, -1)
			g4 += cost(greedy(p4, B, false))
			o4 += cost(greedy(p4, B, true))
			if B <= 32 {
				g16 += cost(greedy(mk(16*B, -1), B, false))
			}
		}
		n := float64(trials)
		uni := E * (1 - math.Pow(1-8.0/E, float64(B)))
		cell := func(x float64) string { return fmt.Sprintf("%5.1f [%5.1f]", x, gb(x)) }
		g16s := "-"
		if B <= 32 {
			g16s = cell(g16 / n)
		}
		fmt.Printf("%4d %10s %16s %16s %16s %16s %16s\n", B, cell(uni), cell(rm/n), cell(st/n), cell(g4/n), cell(o4/n), g16s)
	}
}
