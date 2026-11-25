#!/usr/bin/env python3
"""Minimal GGUF metadata + tensor-table dumper (no deps)."""
import collections
import struct
import sys

GT = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1',
    10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS',
    17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S',
    23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M',
    30: 'BF16', 34: 'TQ1_0', 35: 'TQ2_0', 39: 'MXFP4', 200: 'NV4', 201: 'MX8',
}
SC = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError("truncated GGUF file")
    return data


def rd(f, fmt):
    full = '<' + fmt
    return struct.unpack(full, read_exact(f, struct.calcsize(full)))[0]


def rstr(f):
    n = rd(f, 'Q')
    if n > (1 << 26):
        raise ValueError(f"unreasonable GGUF string length {n}")
    return read_exact(f, n).decode('utf-8', 'replace')


def rval(f, t):
    if t in SC:
        return rd(f, SC[t])
    if t == 8:
        return rstr(f)
    if t == 9:
        et = rd(f, 'I')
        n = rd(f, 'Q')
        if n > (1 << 26):
            raise ValueError(f"unreasonable GGUF array length {n}")
        if et in SC:
            sz = struct.calcsize('<' + SC[et])
            data = read_exact(f, sz * n)
            return list(struct.unpack('<%d%s' % (n, SC[et]), data))
        return [rval(f, et) for _ in range(n)]
    raise ValueError(f"unsupported GGUF metadata type {t}")


def dump_gguf(p, show_tensors=True):
    with open(p, 'rb') as f:
        if read_exact(f, 4) != b'GGUF':
            raise ValueError("expected GGUF magic")
        ver = rd(f, 'I')
        if ver not in (2, 3):
            raise ValueError(f"unsupported GGUF version {ver}")
        nt = rd(f, 'Q')
        nkv = rd(f, 'Q')
        print('version', ver, 'tensors', nt, 'kv', nkv)
        kv = {}
        for _ in range(nkv):
            k = rstr(f)
            t = rd(f, 'I')
            v = rval(f, t)
            kv[k] = v
            if isinstance(v, list) and len(v) > 16:
                print(f'{k} = [list len {len(v)}] first: {v[:8]}')
            elif isinstance(v, str) and len(v) > 300:
                print(f'{k} = [str len {len(v)}]')
            else:
                print(f'{k} = {v}')
        tens = []
        for _ in range(nt):
            name = rstr(f)
            nd = rd(f, 'I')
            if nd < 1 or nd > 4:
                raise ValueError(f"tensor {name}: invalid rank {nd}")
            dims = [rd(f, 'Q') for _ in range(nd)]
            t = rd(f, 'I')
            off = rd(f, 'Q')
            tens.append((name, dims, GT.get(t, t), off))
        if show_tensors:
            seen = collections.OrderedDict()
            for name, dims, t, _ in tens:
                parts = name.split('.')
                key = '.'.join('N' if x.isdigit() else x for x in parts)
                seen.setdefault((key, tuple(dims), t), []).append(name)
            for (key, dims, t), names in seen.items():
                print(f'{key:45s} {str(list(dims)):28s} {t:8s} x{len(names)}  e.g. {names[0]}')
        cnt = collections.Counter(t for _, _, t, _ in tens)
        print(cnt)


def main(argv):
    if len(argv) != 2:
        print("usage: gguf_dump.py <model.gguf>", file=sys.stderr)
        return 2
    try:
        dump_gguf(argv[1])
    except (OSError, EOFError, ValueError, struct.error) as exc:
        print(f"gguf_dump.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
