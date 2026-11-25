# Contributing Without an M5 Mac

`inky`'s inference kernels require macOS with Metal 4 and Apple M5 neural accelerators for live model execution, but all parsers, tokenizers, HTTP protocol translation, paged-KV block bookkeeping, lane scheduling, formal TLA+ specifications, and Go test harnesses can be built and verified on non-M5 hardware.

## What Runs Everywhere vs What Needs an M5

| Machine | Supported Validation |
|---|---|
| **Any Apple Silicon or Intel Mac (no M5 needed)** | Full `cargo build --locked`, `cargo test --locked --bins --tests` (CPU unit tests, HTTP framing/auth/deadline tests, 10,000-mutation GGUF/tokenizer fuzz suite), `cargo publish --locked --dry-run`, all Go `-race` suites, and all TLA+ model & trace-mutation checks. Metal shaders compile at runtime (`newLibraryWithSource`), so compiling and running unit tests requires no M5 GPU or model weights. |
| **Linux (`x86_64` or `aarch64`)** | Workflow/release gate tests (`tests/test_release_ci.py`), all 6 Go `-race` suites (`soak_client`, `tools/load`, `tools/inkconvert`, `tests/fleetcheck` with `bubblewrap`, `tests/servecheck`, `tools/replay`), TLA+ model & trace checks (`scripts/check-tla-tiers.sh`, `scripts/check-tla.sh`), and `cargo-fuzz` against `fuzz/fuzz_targets/gguf_load.rs` (GGUF + BPE tokenizer parser without Metal). |
| **Apple M5 Mac (self-hosted CI: `m5-gpu`, `m5-serve`, `m5-full`)** | Live GPU logit agreement against `llama.cpp` (`tests/ci.sh`), GPU negative-control sabotage gates (`INK_*_SABOTAGE=1`), long poisoned-pool & SSD-spill gates (`tests/nightly.sh`), golden logit hashes (`tests/golden.sh`), and live `inky serve` HTTP/SSE probes (`tests/serve.sh`, `tests/serve-negative.sh`). |

## 1. macOS (Without an M5 GPU or Model File)

```sh
# CPU unit & integration tests (debug assertions + overflow checks + 10K GGUF/tokenizer fuzzer)
RUSTFLAGS="-D warnings" cargo test --locked --bins --tests

# Verify the serve header policy test runs and passes
cargo test --locked --bins serve_header_policy

# Verify crates.io packaging allowlist and metadata
cargo publish --locked --dry-run
```

## 2. Linux or macOS: Go `-race` Tool & Harness Suites

Requires Go `1.26+` (and `bubblewrap` on Linux for `tests/fleetcheck` sandbox tests):

```sh
python3 tests/test_release_ci.py
go test -race tests/soak_client.go tests/soak_client_test.go
(cd tools/load && go test -race ./...)
(cd tools/inkconvert && go test -race ./...)
(cd tests && go vet ./fleetcheck/... && go test -race ./fleetcheck/...)
(cd tests && go vet ./servecheck && go test -race ./servecheck)
(cd tools/replay && go vet ./... && go test -race ./...)
```

## 3. Linux or macOS: TLA+ Models, Reversed Witnesses, and Trace Mutants

Requires Java `21+` (`scripts/check-tla.sh` downloads and verifies the SHA-256-pinned `tla2tools.jar`) and `rustc`:

```sh
# Verify every TLA+ spec, witness, and trace mutant is wired into exactly one CI tier
scripts/check-tla-tiers.sh
scripts/check-tla-tiers.sh --self-test

# Pull-request TLA+ suites (MemMode, BlockLayout/InBlock, GroupedDecode, production traces + mutants)
TLA_JAR_CACHE="$HOME/.cache/inky-tla" scripts/check-tla.sh memmode
TLA_JAR_CACHE="$HOME/.cache/inky-tla" scripts/check-tla.sh blocks
TLA_JAR_CACHE="$HOME/.cache/inky-tla" scripts/check-tla.sh grouped
TLA_JAR_CACHE="$HOME/.cache/inky-tla" scripts/check-tla.sh traces

# Nightly exhaustive models (requires >= 12 GiB RAM for -Xmx8g on scheduler)
TLA_NIGHTLY=1 TLA_JAR_CACHE="$HOME/.cache/inky-tla" scripts/check-tla.sh layout-nightly
TLA_NIGHTLY=1 TLC_WORKERS=4 TLA_JAR_CACHE="$HOME/.cache/inky-tla" scripts/check-tla.sh scheduler
```

## 4. Linux or macOS: `cargo-fuzz` on GGUF & Tokenizer (`fuzz/`)

`fuzz/` is a standalone workspace targeting `src/gguf.rs` and `src/tokenizer.rs` with zero Metal/`objc2` dependencies:

```sh
cargo +nightly fuzz run --fuzz-dir fuzz gguf_load -- -max_total_time=60
```

## 5. M5 Hardware Gates (Maintainer / Self-Hosted Runner Reference)

On an M5 Mac with `Inkling-2-Mini-Q4_K_M.gguf` and reference logits in `~/inky-ci/ref`:

```sh
cargo build --locked --profile ci
INKY_BIN=./target/ci/inky tests/ci.sh pr
INKY_SERVE_PROBES_ONLY=1 tests/serve.sh ./target/ci/inky
INKY_SERVE_PROBES_ONLY=1 tests/serve-negative.sh tests/serve.sh ./target/ci/inky
```

Outside pull requests run the `unit`, `go`, and `tla` jobs automatically on GitHub-hosted runners; maintainer review triggers the self-hosted `m5-gpu` and `m5-serve` jobs before merge.
