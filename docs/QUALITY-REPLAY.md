# Quality Replay, `--kv8`, and Thinking Effort Sweep

This document records the deterministic quality replay suite (`tools/quality` + `tests/quality/cards.jsonl`) and measured results on **Apple M5 Pro (64 GiB)** running `inky serve` with `Inkling-2-Mini-Q4_K_M.gguf` (`-c 16384 -n 4096 --lanes 1`, `temperature=0`, `seed=42`).

## Suite Design (`tools/quality`, `tests/quality/cards.jsonl`)

`tools/quality` is a zero-dependency Go package and CLI that evaluates coding, unified-diff patch, and JSON tool-call quality against `/v1/chat/completions` and inspects `/health` for KV cache width and memory reserve telemetry.

Nine curated cards span three categories (three cards each):

1. **`mechanical-edit`**:
   - `mech-rename-symbol`: Unified diff (`---`/`+++`, `@@` hunks) renaming `Counter.Inc`/`Dec` to `Increment`/`Decrement` across methods and call sites; validated via unified-diff parsing, patch application to `counter.go`, `go/parser` AST symbol checks, and forbidden-symbol checks.
   - `mech-add-field-tag`: Single `edit_file` JSON tool call adding `MaxRetries int` (`json:"max_retries"`) and `MaxRetries: 3` in `DefaultConfig()`; validated via JSON tool-call schema checks, patch application to `config.go`, `go/parser` AST symbol checks, and substring assertions.
   - `mech-extract-const`: Go refactor extracting `500 * time.Millisecond` into `const defaultBackoff`; validated via `go/parser` AST symbols and literal removal assertions.
2. **`bugfix`**:
   - `bugfix-clamp-upper`: `apply_patch` JSON tool call fixing an upper-bound bug in `Clamp`; validated via JSON tool-call schema checks, unified diff parsing + application to `clamp.go`, `go/parser`, and control-flow regex assertions.
   - `bugfix-ring-wrap`: Unified diff fixing `RingBuffer.Push` head advancement and size clamping on full-buffer wraparound; validated via diff application to `ring.go`, `go/parser`, and wraparound regexes.
   - `bugfix-binary-search-overflow`: Fixing `for lo <= hi` and overflow-safe midpoint `(hi - lo)` in `BinarySearch`; validated via `go/parser` and positive/negative logic assertions.
3. **`multi-step-algorithm`**:
   - `algo-lru-cache`: Full Go `LRUCache` using `container/list` and `map[string]*list.Element`; validated via `go/parser` AST method/type checks and O(1) eviction/recency assertions.
   - `algo-topo-sort`: Kahn's topological sort `TopoSort(n int, edges [][2]int) ([]int, bool)` with cycle detection; validated via `go/parser` and cycle/in-degree assertions.
   - `algo-interval-merge-tool`: `submit_solution` JSON tool call containing `package intervals` with `MergeIntervals`; validated via JSON tool-call schema checks, `go/parser`, and `sort.Slice` assertions.

Run the deterministic unit test suite locally or in CI (`< 2s`):

```sh
cd tools/quality && go vet ./... && go test -race -count=1 ./...
```

Run against a live `inky serve` instance:

```sh
cd tools/quality && go run . \
  -cards ../../tests/quality/cards.jsonl \
  -url http://127.0.0.1:18921 \
  -name fp16 \
  -efforts none,low,medium,high \
  -temp 0 -seed 42
```

## Reference `fp16` vs `--kv8` (`t-later-quality`, `t-later-kv8`)

Measured on **Apple M5 Pro (64 GiB)** at `reasoning_effort=medium`, `temperature=0`, `seed=42` (`tests/quality/results_m5_pro.json`):

| Metric | Reference `fp16` | Candidate `--kv8` | Delta / Ratio |
| --- | ---: | ---: | ---: |
| Cards passed | **8/9 (88.9%)** | **9/9 (100.0%)** | +1 card (`mech-add-field-tag`) |
| Check parity rate | — | **88.9% (8/9)** | — |
| Exact output match (`temp=0`) | — | **33.3% (3/9)** | Not bit-exact |
| Mean lexical token agreement (`temp=0`) | — | **57.3%** | Mean prefix match: **42.5%** |
| KV cache element width (`/health`) | `2 B/elem` | `1 B/elem` | **2.0x reduction** |
| Per-lane worst-case reserve (`-c 16384`) | `0.66 GiB` (`712,980,768 B`) | `0.34 GiB` (`360,659,232 B`) | **1.98x saving** |
| Layer 3 paged KV pool (`118 win + 731 full`) | `2.58 GiB` (`1.15 + 1.43`) | `1.29 GiB` (`0.58 + 0.71`) | **2.00x reduction** |
| Avg wall latency per card | `38,745.0 ms` | `32,252.1 ms` | `-16.8%` |
| Avg decode time (`X-Inky-Decode-Ms`) | `38,204.0 ms` | `31,735.0 ms` | `-16.9%` |
| Total completion tokens (9 cards) | `7,973` | `7,399` | `-7.2%` |

Per-card `fp16` vs `--kv8` breakdown at `temp=0`, `seed=42`:

| Card | Kind | `fp16` | `--kv8` | Exact Match | Lexical Token Agreement | `fp16` Wall (ms) | `--kv8` Wall (ms) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `mech-rename-symbol` | `mechanical-edit` | PASS | PASS | true | 100.0% | 36,433.1 | 36,042.5 |
| `mech-add-field-tag` | `mechanical-edit` | FAIL | PASS | false | 0.0% | 83,320.3 | 35,818.6 |
| `mech-extract-const` | `mechanical-edit` | PASS | PASS | true | 100.0% | 14,444.5 | 12,003.5 |
| `bugfix-clamp-upper` | `bugfix` | PASS | PASS | false | 34.8% | 25,313.5 | 14,407.5 |
| `bugfix-ring-wrap` | `bugfix` | PASS | PASS | false | 79.4% | 41,161.2 | 44,046.1 |
| `bugfix-binary-search-overflow` | `bugfix` | PASS | PASS | true | 100.0% | 20,719.2 | 16,166.9 |
| `algo-lru-cache` | `multi-step-algorithm` | PASS | PASS | false | 64.5% | 47,414.0 | 52,261.6 |
| `algo-topo-sort` | `multi-step-algorithm` | PASS | PASS | false | 16.4% | 44,150.1 | 44,327.9 |
| `algo-interval-merge-tool` | `multi-step-algorithm` | PASS | PASS | false | 21.1% | 35,749.1 | 35,194.6 |

**Takeaway (`t-later-kv8`)**: `--kv8` cuts KV element storage and Layer 3 paged KV pool footprint by **2.00x** (`2.58 GiB` -> `1.29 GiB` at `118` window + `731` full slots; `0.66 GiB` -> `0.34 GiB` per-lane reserve). As documented in `inky`, `--kv8` is not bit-exact (`33.3%` byte-exact match and `57.3%` lexical token agreement at `temp=0`), yet preserves full coding/diff/tool-call functionality (`9/9` pass vs `8/9` in `fp16-medium`).

## Thinking Effort Sweep (`t-later-effort`)

Measured on **Apple M5 Pro (64 GiB)** (`fp16`, `temperature=0`, `seed=42`) across `reasoning_effort` = `none`, `low`, `medium`, `high`:

| Effort | Card Kind | Passed | Pass Rate | Total Gen Tokens | Avg Gen Tokens | Avg Wall (ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `none` | `mechanical-edit` | 2/3 | 66.7% | 2,920 | 973.3 | 38,244.4 |
| `none` | `bugfix` | 3/3 | 100.0% | 1,238 | 412.7 | 16,209.3 |
| `none` | `multi-step-algorithm` | 3/3 | 100.0% | 3,440 | 1,146.7 | 44,848.1 |
| `low` | `mechanical-edit` | **3/3** | **100.0%** | **2,005** | **668.3** | **26,204.2** |
| `low` | `bugfix` | **3/3** | **100.0%** | **1,059** | **353.0** | **13,903.6** |
| `low` | `multi-step-algorithm` | **3/3** | **100.0%** | **2,468** | **822.7** | **41,994.3** |
| `medium` | `mechanical-edit` | 2/3 | 66.7% | 2,527 | 842.3 | 44,732.7 |
| `medium` | `bugfix` | 3/3 | 100.0% | 2,225 | 741.7 | 29,064.6 |
| `medium` | `multi-step-algorithm` | 3/3 | 100.0% | 3,221 | 1,073.7 | 42,437.7 |
| `high` | `mechanical-edit` | 3/3 | 100.0% | 2,145 | 715.0 | 27,938.8 |
| `high` | `bugfix` | 2/3 | 66.7% | 2,861 | 953.7 | 37,260.4 |
| `high` | `multi-step-algorithm` | 3/3 | 100.0% | 4,423 | 1,474.3 | 57,592.1 |

Overall totals across all 9 cards by effort:

| Effort | Overall Passed | Total Gen Tokens | Avg Wall (ms) |
| --- | ---: | ---: | ---: |
| `none` | 8/9 (88.9%) | 7,598 | 33,100.6 |
| `low` | **9/9 (100.0%)** | **5,532** | **27,367.4** |
| `medium` | 8/9 (88.9%) | 7,973 | 38,745.0 |
| `high` | 8/9 (88.9%) | 9,429 | 40,930.4 |

### Lowest Passing Effort per Card Kind

- **`mechanical-edit`**: **`low`** (`3/3`, avg `668.3` tokens, `26,204.2 ms`). At `none`, `mech-add-field-tag` exhausted its `1536`-token budget deliberating over tool-call ordering.
- **`bugfix`**: **`none`** (`3/3`, avg `412.7` tokens, `16,209.3 ms`). Notably, **`low`** still passes `3/3` while using **14.5% fewer tokens** (`353.0` vs `412.7`) and lower wall time (`13,903.6 ms`), whereas `high` over-deliberated `bugfix-ring-wrap` past its `1536`-token cap.
- **`multi-step-algorithm`**: **`none`** (`3/3`, avg `1,146.7` tokens, `44,848.1 ms`). Again **`low`** is the Pareto optimum, passing `3/3` with **28.3% fewer tokens** than `none` (`822.7` vs `1,146.7`) and **44.2% fewer tokens** than `high` (`1,474.3`).

Overall, **`reasoning_effort=low`** is the lowest effort that achieves **9/9 (100.0%)** across all three card kinds while minimizing total generated tokens (`5,532`) and average latency (`27,367.4 ms`).
