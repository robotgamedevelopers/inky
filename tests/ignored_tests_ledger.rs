use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).expect("read src dir");
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn extract_ignored_tests(root: &Path) -> BTreeMap<(String, String), String> {
    let mut files = Vec::new();
    collect_rs_files(&root.join("src"), &mut files);
    files.sort();
    let mut found = BTreeMap::new();
    for file in files {
        let rel = file.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        let content = fs::read_to_string(&file).expect("read rust source");
        let lines: Vec<&str> = content.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("#[ignore") {
                let mut fn_name = None;
                for next in lines.iter().skip(idx + 1).take(8) {
                    let nt = next.trim();
                    if let Some(rest) = nt.strip_prefix("fn ") {
                        let name: String =
                            rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
                        if !name.is_empty() {
                            fn_name = Some(name);
                            break;
                        }
                    }
                }
                let name =
                    fn_name.unwrap_or_else(|| panic!("could not find `fn` following #[ignore] at {rel}:{}", idx + 1));
                found.insert((rel.clone(), name), trimmed.to_string());
            }
        }
    }
    found
}

#[test]
fn every_ignored_test_is_ledgered_and_wired_to_a_ci_tier() {
    let root = repo_root();
    let found = extract_ignored_tests(&root);

    // Canonical ledger of every `#[ignore]` test in `src/`: `(file, fn_name, runner_file, needle_in_runner)`.
    let ledger: &[(&str, &str, &str, &str)] = &[
        (
            "src/blocks.rs",
            "gpu_pool_zero_fills_a_taken_slot",
            ".github/workflows/nightly.yml",
            "blocks::tests::gpu_pool_zero_fills_a_taken_slot",
        ),
        (
            "src/blocks.rs",
            "metal_in_slot_formula_matches_key_row_and_inblock_trace_across_full_and_window",
            ".github/workflows/nightly.yml",
            "blocks::tests::metal_in_slot_formula_matches_key_row_and_inblock_trace_across_full_and_window",
        ),
        (
            "src/cache_scheduler/tests.rs",
            "dump_block_layout",
            ".github/workflows/nightly.yml",
            "cache_scheduler::tests::dump_block_layout",
        ),
        (
            "src/cache_scheduler/tests.rs",
            "dump_cache_scheduler",
            ".github/workflows/nightly.yml",
            "cache_scheduler::tests::dump_cache_scheduler",
        ),
        (
            "src/cache_scheduler/tests.rs",
            "walk_cache_scheduler_propagates_trace_errors_on_every_action_and_reaches_drop_slide_spill",
            ".github/workflows/nightly.yml",
            "cache_scheduler::tests::walk_cache_scheduler_propagates_trace_errors_on_every_action_and_reaches_drop_slide_spill",
        ),
        (
            "src/engine/tests.rs",
            "debug_single_token_decode_native_checkpoint",
            "tests/nightly.sh",
            "engine::tests::debug_single_token_decode_native_checkpoint",
        ),
        (
            "src/engine/tests.rs",
            "disk_tier_matches_resident_logits_across_32k_staging_resize_transition",
            "tests/nightly.sh",
            "engine::tests::disk_tier_matches_resident_logits_across_32k_staging_resize_transition",
        ),
        (
            "src/engine/tests.rs",
            "dspark_gpu_guards_and_rejection_sampling_gate",
            ".github/workflows/nightly.yml",
            "engine::tests::dspark_gpu_guards_and_rejection_sampling_gate",
        ),
        (
            "src/engine/tests.rs",
            "engine_encode_decode_64_positions_syncs_once_then_63_zero_steps",
            ".github/workflows/nightly.yml",
            "engine::tests::engine_encode_decode_64_positions_syncs_once_then_63_zero_steps",
        ),
        (
            "src/engine/tests.rs",
            "full_attn_budget_default_bit_identical_and_pruned_4k_coherent_on_m5",
            "tests/nightly.sh",
            "engine::tests::full_attn_budget_default_bit_identical_and_pruned_4k_coherent_on_m5",
        ),
        (
            "src/engine/tests.rs",
            "k_attn_b_and_blk_are_split_invariant_across_sliding_window",
            ".github/workflows/nightly.yml",
            "engine::tests::k_attn_b_and_blk_are_split_invariant_across_sliding_window",
        ),
        (
            "src/engine/tests.rs",
            "loopcd_mixed_batch_preserves_disabled_rows",
            "tests/loopcd-gate.sh",
            "engine::tests::loopcd_mixed_batch_preserves_disabled_rows",
        ),
        (
            "src/engine/tests.rs",
            "metal_shader_declarations_and_compiler_validation",
            ".github/workflows/nightly.yml",
            "engine::tests::metal_shader_declarations_and_compiler_validation",
        ),
        (
            "src/engine/tests.rs",
            "paged_verify_kv_snapshot_and_restore_covers_all_48_loop_passes",
            ".github/workflows/nightly.yml",
            "engine::tests::paged_verify_kv_snapshot_and_restore_covers_all_48_loop_passes",
        ),
        (
            "src/engine/tests.rs",
            "private_prefill_clears_the_last_window_tile_between_lanes",
            "tests/ci.sh",
            "engine::tests::private_prefill_clears_the_last_window_tile_between_lanes",
        ),
        (
            "src/engine/tests.rs",
            "router_handles_nonfinite_scores",
            ".github/workflows/nightly.yml",
            "engine::tests::router_handles_nonfinite_scores",
        ),
        (
            "src/engine/tests.rs",
            "sub4bit_quant_metal_kernels_match_cpu_and_save_bytes",
            "tests/nightly.sh",
            "engine::tests::sub4bit_quant_metal_kernels_match_cpu_and_save_bytes",
        ),
        (
            "src/engine/tests.rs",
            "sub4bit_quant_real_model_lmhead_logits_cosine_and_top1_on_m5",
            "tests/nightly.sh",
            "engine::tests::sub4bit_quant_real_model_lmhead_logits_cosine_and_top1_on_m5",
        ),
        (
            "src/gguf.rs",
            "multi_layer_native_checkpoint_gguf_loads_nv4_gscale_and_mx8",
            ".github/workflows/nightly.yml",
            "gguf::tests::multi_layer_native_checkpoint_gguf_loads_nv4_gscale_and_mx8",
        ),
        (
            "src/loopcd_tests.rs",
            "loopcd_cached_prompt_first_token_and_mixed_tie_isolation",
            "tests/loopcd-gate.sh",
            "loopcd_tests::loopcd_cached_prompt_first_token_and_mixed_tie_isolation",
        ),
        (
            "src/loopcd_tests.rs",
            "loopcd_cli_first_token",
            "tests/loopcd-gate.sh",
            "loopcd_tests::loopcd_cli_first_token",
        ),
        (
            "src/loopcd_tests.rs",
            "loopcd_paged_mixed_lane_state",
            "tests/loopcd-gate.sh",
            "loopcd_tests::loopcd_paged_mixed_lane_state",
        ),
        (
            "src/memd.rs",
            "fake_socket_rebinds_with_private_mode_and_reports_its_peer_uid",
            ".github/workflows/nightly.yml",
            "memd::tests:: -- --ignored",
        ),
        (
            "src/memd.rs",
            "legacy_cleanup_removes_the_socket_and_retries_after_a_directory_error",
            ".github/workflows/nightly.yml",
            "memd::tests:: -- --ignored",
        ),
        ("src/prefix.rs", "prefix_trace_dump", "specs/run_tlc.zsh", "prefix::tests::prefix_trace_dump"),
        (
            "src/sampler.rs",
            "residual_emits_target_argmax_outside_draft_top64_softcap_after_topk",
            ".github/workflows/nightly.yml",
            "sampler::tests::residual_emits_target_argmax_outside_draft_top64_softcap_after_topk",
        ),
        (
            "src/scheduler/tests.rs",
            "bimodal_2k_64k_tight_budget_with_mid_step_disconnects",
            ".github/workflows/nightly.yml",
            "scheduler::tests::bimodal_2k_64k_tight_budget_with_mid_step_disconnects",
        ),
        (
            "src/serve.rs",
            "gguf_embedded_chat_template_matches_serve_prompt_render_token_for_token",
            ".github/workflows/nightly.yml",
            "serve::tests::gguf_embedded_chat_template_matches_serve_prompt_render_token_for_token",
        ),
        (
            "src/serve.rs",
            "overlap_host_work_drains_rx_preserves_cancel_in_step_pins_and_maintains_spill",
            ".github/workflows/nightly.yml",
            "serve::tests::overlap_host_work_drains_rx_preserves_cancel_in_step_pins_and_maintains_spill",
        ),
        (
            "src/serve.rs",
            "serve_layer3_three_turn_10k_cached_matches_cold_solo",
            "tests/nightly.sh",
            "serve::tests::serve_layer3_three_turn_10k_cached_matches_cold_solo",
        ),
        (
            "src/serve/block_cache.rs",
            "long_spill_restore_keeps_source_and_accepts_early_fork",
            ".github/workflows/nightly.yml",
            "serve::block_cache::tests::long_spill_restore_keeps_source_and_accepts_early_fork",
        ),
        (
            "src/serve/block_cache.rs",
            "maintain_spill_during_gpu_trims_excess_spilled_starts_without_touching_pools",
            ".github/workflows/nightly.yml",
            "serve::block_cache::tests::maintain_spill_during_gpu_trims_excess_spilled_starts_without_touching_pools",
        ),
        (
            "src/serve/block_cache.rs",
            "spill_index_persists_across_reopen_and_async_restore_keeps_decoding_responsive",
            ".github/workflows/nightly.yml",
            "serve::block_cache::tests::spill_index_persists_across_reopen_and_async_restore_keeps_decoding_responsive",
        ),
        (
            "src/serve/block_cache.rs",
            "spill_index_rejects_full_attn_budget_and_model_fingerprint_mismatches",
            ".github/workflows/nightly.yml",
            "serve::block_cache::tests::spill_index_rejects_full_attn_budget_and_model_fingerprint_mismatches",
        ),
        (
            "src/serve/block_cache.rs",
            "spill_read_failure_releases_provisional_blocks_and_preserves_resident_match",
            ".github/workflows/nightly.yml",
            "serve::block_cache::tests::spill_read_failure_releases_provisional_blocks_and_preserves_resident_match",
        ),
        (
            "src/serve/block_cache.rs",
            "spill_retained_start_owns_reused_disk_blocks_across_trim_and_same_tip_overwrite",
            ".github/workflows/nightly.yml",
            "serve::block_cache::tests::spill_retained_start_owns_reused_disk_blocks_across_trim_and_same_tip_overwrite",
        ),
        (
            "src/serve/block_cache.rs",
            "spill_store_eviction_restore_and_corrupt_fallback",
            ".github/workflows/nightly.yml",
            "serve::block_cache::tests::spill_store_eviction_restore_and_corrupt_fallback",
        ),
        (
            "src/serve/block_cache/admission_tests.rs",
            "burst_32_concurrent_65k_prompts_admits_in_under_2ms_with_exact_reference_parity",
            ".github/workflows/nightly.yml",
            "serve::block_cache::admission_tests::burst_32_concurrent_65k_prompts_admits_in_under_2ms_with_exact_reference_parity",
        ),
        (
            "src/serve/block_cache/tests/spill_model.rs",
            "restored_long_prefix_and_early_fork_match_cold_logits",
            "tests/nightly.sh",
            "serve::block_cache::tests::spill_model::restored_long_prefix_and_early_fork_match_cold_logits",
        ),
        ("src/serve/step_trace.rs", "dump", ".github/workflows/nightly.yml", "serve::scheduler_trace::dump"),
        (
            "src/session.rs",
            "cli_session_kv_disk_recovers_across_tier_entry_and_active_tier_rollback",
            "tests/nightly.sh",
            "session::tests::cli_session_kv_disk_recovers_across_tier_entry_and_active_tier_rollback",
        ),
        (
            "src/session.rs",
            "cli_session_recovers_from_injected_prefill_and_decode_failures",
            "tests/nightly.sh",
            "session::tests::cli_session_recovers_from_injected_prefill_and_decode_failures",
        ),
        (
            "src/spill/tests.rs",
            "kill9_mid_write_loses_cache_never_correctness",
            ".github/workflows/nightly.yml",
            "spill::tests::kill9_mid_write_loses_cache_never_correctness",
        ),
        ("src/spill/tests.rs", "kill9_writer_helper", "src/spill/tests.rs", "spill::tests::kill9_writer_helper"),
        (
            "src/tokenizer.rs",
            "real_gguf_tokenizer_74k_corpus_roundtrip_and_determinism",
            ".github/workflows/nightly.yml",
            "tokenizer::tests::real_gguf_tokenizer_74k_corpus_roundtrip_and_determinism",
        ),
    ];

    let expected_keys: BTreeSet<(String, String)> =
        ledger.iter().map(|(f, n, _, _)| ((*f).to_string(), (*n).to_string())).collect();
    let actual_keys: BTreeSet<(String, String)> = found.keys().cloned().collect();
    assert_eq!(
        actual_keys, expected_keys,
        "every #[ignore] test in src/ must appear in tests/ignored_tests_ledger.rs and run in a CI tier"
    );

    for &(src_file, fn_name, runner, needle) in ledger {
        let runner_text = fs::read_to_string(root.join(runner))
            .unwrap_or_else(|e| panic!("failed to read tier runner {runner} for {src_file}::{fn_name}: {e}"));
        assert!(
            runner_text.contains(needle),
            "ignored test {src_file}::{fn_name} missing `{needle}` in tier runner {runner}"
        );
    }
}

#[test]
fn workflow_cargo_commands_use_locked_and_filtered_tests_assert_non_zero_pass_count() {
    let root = repo_root();
    for wf in [".github/workflows/ci.yml", ".github/workflows/nightly.yml"] {
        let content = fs::read_to_string(root.join(wf)).expect("read workflow");
        let lines: Vec<&str> = content.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with('#') {
                continue;
            }
            for sub in ["cargo build", "cargo test", "cargo publish"] {
                if let Some(pos) = trimmed.find(sub) {
                    let tail = &trimmed[pos..];
                    assert!(
                        tail.contains("--locked"),
                        "{wf}:{} `{sub}` invocation missing --locked: {trimmed}",
                        idx + 1
                    );
                }
            }
            // Any filtered `cargo test` (anything other than full `--bins --tests`) must assert `[1-9][0-9]* passed; 0 failed;` nearby.
            if let Some(pos) = trimmed.find("cargo test") {
                let tail = &trimmed[pos..];
                if !tail.contains("--bins --tests") {
                    let window_end = (idx + 6).min(lines.len());
                    let window = lines[idx..window_end].join("\n");
                    assert!(
                        window.contains("passed; 0 failed;"),
                        "{wf}:{} filtered cargo test lacks non-zero `passed; 0 failed;` assertion:\n{window}",
                        idx + 1
                    );
                }
            }
        }
    }
}
