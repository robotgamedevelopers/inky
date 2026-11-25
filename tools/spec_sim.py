#!/usr/bin/env python3
"""Simulate prompt-lookup speculative decoding with a measured pass-cost curve (bytes model)."""
import sys

COST = {1: 1.0, 2: 1.34, 3: 1.61, 4: 1.85, 5: 2.05, 6: 2.23, 7: 2.40, 8: 2.57}


def load(p):
    P, G = [], []
    with open(p, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            a = line.split()
            if not a:
                continue
            if len(a) < 2 or a[0] not in ('P', 'G'):
                raise ValueError(f"{p}:{line_no}: expected 'P <tok>' or 'G <tok> ...'")
            (P if a[0] == 'P' else G).append(int(a[1]))
    if not G:
        raise ValueError(f"{p}: no generated (G) tokens found")
    return P, G


def main(argv):
    if len(argv) < 2:
        print("usage: spec_sim.py <routes-file>...", file=sys.stderr)
        return 2
    for p in argv[1:]:
        try:
            P, G = load(p)
        except (OSError, ValueError) as exc:
            print(f"spec_sim.py: {exc}", file=sys.stderr)
            return 1
        n = len(G)
        best = None
        for minm in (1, 2, 3, 4, 6):
            for K in (2, 3, 4, 7):
                ctx = list(P)
                i = 0
                cost = 0.0
                while i < n:
                    draft = []
                    for ng in (8, 6, 4, 3, 2, 1):
                        if ng < minm or len(ctx) < ng:
                            continue
                        key = ctx[-ng:]
                        for j in range(len(ctx) - ng - 1, -1, -1):
                            if ctx[j:j + ng] == key:
                                draft = ctx[j + ng:j + ng + K]
                                break
                        if draft:
                            break
                    a = 0
                    while a < len(draft) and i + a < n and draft[a] == G[i + a]:
                        a += 1
                    cost += COST[len(draft) + 1]
                    step = min(a + 1, n - i)
                    ctx += G[i:i + step]
                    i += step
                sp = n / cost
                if best is None or sp > best[0]:
                    best = (sp, minm, K)
        print(f"{p}: best prompt-lookup speedup {best[0]:.2f}x (min match {best[1]}, max draft {best[2]})")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
