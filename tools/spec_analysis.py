#!/usr/bin/env python3
"""Speculative-decoding feasibility: expert-union growth over consecutive tokens + prompt-lookup draft acceptance."""
import sys

LAYERS = 48
EXPERTS = 128
GU, DN = 2.986, 1.66  # MB per expert (gate_up, avg down)
SHARED = 1260.0       # MB per pass: attention/ctrl/router weights + lm_head


def load(p):
    P, G, R = [], [], []
    with open(p, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == 'P':
                if len(parts) < 2:
                    raise ValueError(f"{p}:{line_no}: malformed P record")
                P.append(int(parts[1]))
            elif tag == 'G':
                if len(parts) < 3:
                    raise ValueError(f"{p}:{line_no}: malformed G record")
                G.append(int(parts[1]))
                passes = parts[2].split(';')
                if len(passes) != LAYERS:
                    raise ValueError(f"{p}:{line_no}: expected {LAYERS} layer passes, got {len(passes)}")
                layer_sets = []
                for l_idx, seg in enumerate(passes):
                    if not seg:
                        raise ValueError(f"{p}:{line_no}: empty expert list at layer {l_idx}")
                    exps = {int(e) for e in seg.split(',')}
                    if any(e < 0 or e >= EXPERTS for e in exps):
                        raise ValueError(f"{p}:{line_no}: expert id out of range 0..{EXPERTS - 1}")
                    layer_sets.append(exps)
                R.append(layer_sets)
            else:
                raise ValueError(f"{p}:{line_no}: unknown record tag {tag!r}")
    if not G:
        raise ValueError(f"{p}: no generated (G) tokens found")
    return P, G, R


def main(argv):
    if len(argv) < 2:
        print("usage: spec_analysis.py <routes-file>...", file=sys.stderr)
        return 2
    for p in argv[1:]:
        try:
            P, G, R = load(p)
        except (OSError, ValueError) as exc:
            print(f"spec_analysis.py: {exc}", file=sys.stderr)
            return 1
        n = len(G)
        print(f"== {p}: prompt {len(P)} tok, generated {n}")
        for B in (2, 3, 4, 6, 8):
            if n < B:
                continue
            us = [
                sum(len(set().union(*[R[i + j][l] for j in range(B)])) for l in range(LAYERS)) / LAYERS
                for i in range(0, n - B + 1)
            ]
            u = sum(us) / len(us)
            cost = SHARED + LAYERS * u * (GU + DN)
            single = SHARED + LAYERS * 8 * (GU + DN)
            print(f"  B={B}: mean experts/layer {u:5.1f} (random {EXPERTS*(1-(1-8/EXPERTS)**B):5.1f}) -> pass cost {cost/single:4.2f}x a single-token step")
        # prompt-lookup drafting: longest suffix match (n=2..4) against all earlier tokens, propose up to K
        for K in (3, 5, 8):
            acc_tokens, passes, i = 0, 0, 0
            ctx = list(P)
            while i < n:
                draft = []
                for ng in (4, 3, 2):
                    if len(ctx) < ng:
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
                step = min(a + 1, n - i)          # accepted drafts + the verified token
                ctx += G[i:i + step]
                i += step
                acc_tokens += step
                passes += 1
            print(f"  prompt-lookup K={K}: {acc_tokens/passes:4.2f} tokens per verification pass")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
