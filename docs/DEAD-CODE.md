# Dead-Code Disposition Ledger

Every `allow(dead_code)` attribute in `src/` is audited against two independent checks:

1. **Check 1 (`main`):** `git grep -w <symbol> origin/main -- src/ tests/ scripts/ tools/` plus `RUSTFLAGS="-D warnings" cargo test --locked --bins --tests`.
2. **Check 2 (open pull requests):** `gh pr diff` / `git diff origin/main...<pr>` across all open pull requests on `robotgamedevelopers/inky` (`#42`, `#86`, `#90`, `#103`, `#104`, and peers).

`tests/dead_code_ledger.rs` parses every `allow(dead_code)` attribute in `src/**/*.rs` and enforces exact set equality with the **Retained `allow(dead_code)` Attributes** table below so no unexplained `#[allow(dead_code)]` can drift into the repository.

## Retained `allow(dead_code)` Attributes

<!-- DEAD_CODE_LEDGER_START -->
| File | Target | Category | Reason |
|---|---|---|---|
| `src/blocks.rs` | `#![allow(dead_code)]` | `trace-reference` | Compiled standalone (`--cfg standalone_blocks`) by `scripts/blockbook-traces.rs`, `scripts/cache-traces.rs`, and `scripts/serve-ownership-traces.rs`, and hosts `InBlock` trace helpers (`key_row_at`). |
| `src/cache_scheduler.rs` | `#![allow(dead_code)]` | `trace-reference` | Standalone Layer 4 reference scheduler bound to `specs/CacheScheduler.tla` by `scripts/check-cache-traces.sh`. |
| `src/cache_trace.rs` | `#![allow(dead_code)]` | `trace-reference` | Standalone TLA+ trace emitter consumed by `scripts/cache-traces.rs`. |
| `src/scheduler.rs` | `#![allow(dead_code)]` | `trace-reference` | Layer 4 `ServeStepPlanner` and TLA+ reference helpers bound to `specs/CacheScheduler.tla` by `scripts/check-scheduler-traces.sh`. |
| `src/spill.rs` | `#![allow(dead_code)]` | `trace-reference` | Layer 4 `SpillStore` and `SpillPoolBackend` working-set reservation API consumed by `inky serve`, unit tests, and `scripts/serve-ownership-traces.rs`. |
| `src/main.rs` | `mod blockbook;` | `trace-reference` | `BlockBook` reference model consumed by `scripts/blockbook-traces.rs` and `scripts/cache-traces.rs` (`EVERY` and `PROMPT` are live in `serve::block_cache`). |
| `src/gguf.rs` | `AF(Vec<f64>),` | `kept-spec` | GGUF v3 binary metadata array variant populated by `read_val` (`ty == 12`). |
| `src/engine.rs` | `pub fn no_grp_dec(&self) -> bool { self.no_grp_dec.get() }` | `kept-api` | Symmetric accessor paired with `Engine::set_no_grp_dec`. |
| `src/engine.rs` | `struct DSparkSpanDesc {` | `held-open-pr` | GPU descriptor for `k_dspark_reject`, exercised by `dspark_gpu_guards_and_rejection_sampling_gate` and held for DSpark speculative serving (`#42`, `#86`). |
| `src/engine.rs` | `struct DSparkRejectArgs {` | `held-open-pr` | Uniform argument block for `k_dspark_reject`, exercised by GPU tests and held for DSpark speculative serving (`#42`, `#86`). |
| `src/engine.rs` | `pub struct SpeculativeVerifySpan<'a> {` | `held-open-pr` | Per-span input to `Engine::verify_speculative_gpu`, exercised by GPU tests and held for DSpark speculative serving (`#42`, `#86`). |
| `src/engine.rs` | `pub struct SpeculativeVerifyOut {` | `held-open-pr` | Accepted/resampled output of `Engine::verify_speculative_gpu`, exercised by GPU tests and held for DSpark speculative serving (`#42`, `#86`). |
| `src/engine.rs` | `pub fn reject_sample_gpu(` | `held-open-pr` | GPU `k_dspark_reject` dispatch primitive, exercised by GPU tests and held for DSpark speculative serving (`#42`, `#86`). |
| `src/engine.rs` | `pub fn verify_speculative_gpu(` | `held-open-pr` | Fallible end-to-end GPU speculative verification and rollback entry point, exercised by GPU tests and held for DSpark speculative serving (`#42`, `#86`). |
| `src/engine.rs` | `pub fn loopcd_blend_row(z3: &mut [f32], z1: &[f32], softcap: f32, omega_max: f32) -> f32 {` | `held-open-pr` | Pure-Rust reference mirror of `k_loopcd_blend` (`#[cfg_attr(not(test), allow(dead_code))]`), exercised by unit tests and LoopCD GPU gates. |
| `src/sampler.rs` | `pub fn reject_one(&mut self, draft_tok: u32, target: &[f32], draft: &[f32]) -> u32 {` | `held-open-pr` | Speculative verification primitive (`#[cfg_attr(not(test), allow(dead_code))]`) held for open PRs `#42` and `#86`. |
| `src/sampler.rs` | `pub fn verify_round(&mut self, draft_tokens: &[u32], target_rows: &[&[f32]], draft_rows: &[&[f32]]) -> Vec<u32> {` | `held-open-pr` | Speculative round verification primitive (`#[cfg_attr(not(test), allow(dead_code))]`) held for open PRs `#42` and `#86`. |
| `src/serve.rs` | `draft_sampler: Sampler,` | `held-open-pr` | Per-request independent speculative draft RNG stream initialized once at admission, held for open PRs `#42` and `#86`. |
| `src/serve/block_cache.rs` | `fn evict_start(&mut self, pools: &mut BlockPools, sid: SealedId) -> bool {` | `held-open-pr` | Direct checkpoint eviction entry point exercised by unit tests and open PR `#103` (`src/serve/block_cache/trace.rs`). |
<!-- DEAD_CODE_LEDGER_END -->

## Dispositions Completed in This Audit

- **Stale `#[allow(dead_code)]` attributes stripped on live `--bins` items (10 sites):**
  - `src/main.rs`: `mod prefix` (live in `serve::block_cache` and `poolgate`).
  - `src/engine.rs`: `GumbelSpan`, `Lane::block_table`, `Engine::block_pool`, `Engine::encode_gumbel_spans`.
  - `src/poolgate.rs`: `table_slots`, `Layer3Pool::candidate_label`.
  - `src/serve/block_cache.rs`: `SpilledStart::reasons`, `ServeBlockCache::record_step_deferred_pool_pressure`, `ServeBlockCache::stats`.
- **Test-only and standalone-trace helpers gated under `#[cfg(test)]` / `#[cfg(any(test, standalone_blocks))]` (zero `--bins` `allow(dead_code)`):**
  - `src/engine.rs`: `AttnBlkDecArgs` (`#[cfg(test)]`).
  - `src/prefix.rs`: `BlockHandles::entries`, `SlotRecords`, `LiveError`, `LiveTree` (`#[cfg(test)]`); `PrefixTree::entries`, `PrefixTree::pin_count` (`#[cfg(any(test, standalone_blocks))]`).
  - `src/record.rs`: `parse_jsonl`, `group_by_card` (held in `#[cfg(test)]` for PR `#90`), `Recorder::new` (`#[cfg(test)]`).
  - `src/serve.rs`: `shared_prefix`, `reusable_prefix`, `LaneCand`, `Admission`, `choose_lane` (`#[cfg(test)]`).
  - `src/serve/block_cache.rs`: `spill_store`, `spilled_start_count`, `retained_start_count` (`#[cfg(any(test, standalone_blocks))]`).
- **Confirmed dead items removed after two-check verification across `main` and all open PRs:**
  - `src/serve.rs`: `decode_overwrites_prompt` and its unit test.
  - `src/record.rs`: `card_id_from_headers` (and its 3 test lines; `serve` sanitizes `X-Inky-Card-Id` via `sanitize_card_id`) and `Recorder::with_capacity`.
  - `src/engine.rs`: `Lane::is_blocked`, `Lane::fill_block`, `Lane::attn_key_offset`, `Engine::per_lane_worst_case_bytes`.
  - `src/prefix.rs`: `SealedId::index`.
  - `src/poolgate.rs`: `slot_region` (and its unit test) and `Layer2::candidate_label`.
  - `src/serve/block_cache.rs`: `ServeBlockCache::set_spill_store` and `ServeBlockCache::spill_store_mut`.
- **Cross-lane prefix copy documentation aligned with code (`Engine::copy_lane_prefix` / `inky xcopy`):**
  - `Engine::copy_lane_prefix` remains live for the standalone Layer 2 `inky xcopy` and `inky temp1` gates (and open PR `#42`/`#104`), while `src/serve.rs`, `src/main.rs` (`xcopy`), and `docs/README.md` accurately state that `inky serve` shares prefixes zero-copy through Layer 3 `ServeBlockCache` rather than calling `copy_lane_prefix`.
