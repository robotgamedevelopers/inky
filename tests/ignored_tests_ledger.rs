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
        let rel = file
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let content = fs::read_to_string(&file).expect("read rust source");
        let lines: Vec<&str> = content.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("#[ignore") {
                let mut fn_name = None;
                for next in lines.iter().skip(idx + 1).take(8) {
                    let nt = next.trim();
                    if let Some(rest) = nt.strip_prefix("fn ") {
                        let name: String = rest
                            .chars()
                            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                            .collect();
                        if !name.is_empty() {
                            fn_name = Some(name);
                            break;
                        }
                    }
                }
                let name = fn_name.unwrap_or_else(|| {
                    panic!("could not find `fn` following #[ignore] at {rel}:{}", idx + 1)
                });
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
            ".github/workflows/ci.yml",
            "blocks::tests::gpu_pool_zero_fills_a_taken_slot",
        ),
        (
            "src/cache_scheduler/tests.rs",
            "dump_block_layout",
            ".github/workflows/ci.yml",
            "cache_scheduler::tests::dump_block_layout",
        ),
        (
            "src/cache_scheduler/tests.rs",
            "dump_cache_scheduler",
            ".github/workflows/ci.yml",
            "cache_scheduler::tests::dump_cache_scheduler",
        ),
        (
            "src/engine/tests.rs",
            "dspark_gpu_guards_and_rejection_sampling_gate",
            ".github/workflows/ci.yml",
            "engine::tests::dspark_gpu_guards_and_rejection_sampling_gate",
        ),
        (
            "src/engine/tests.rs",
            "loopcd_mixed_batch_preserves_disabled_rows",
            ".github/workflows/nightly.yml",
            "loopcd_ -- --ignored",
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
            ".github/workflows/ci.yml",
            "engine::tests::router_handles_nonfinite_scores",
        ),
        (
            "src/loopcd_tests.rs",
            "loopcd_model_first_token_and_mixed_lanes",
            ".github/workflows/nightly.yml",
            "loopcd_ -- --ignored",
        ),
        (
            "src/memd.rs",
            "fake_socket_rebinds_with_private_mode_and_reports_its_peer_uid",
            ".github/workflows/ci.yml",
            "memd::tests:: -- --ignored",
        ),
        (
            "src/memd.rs",
            "legacy_cleanup_removes_the_socket_and_retries_after_a_directory_error",
            ".github/workflows/ci.yml",
            "memd::tests:: -- --ignored",
        ),
        (
            "src/prefix.rs",
            "prefix_trace_dump",
            "specs/run_tlc.zsh",
            "prefix::tests::prefix_trace_dump",
        ),
        (
            "src/serve.rs",
            "serve_layer3_three_turn_10k_cached_matches_cold_solo",
            "tests/nightly.sh",
            "serve::tests::serve_layer3_three_turn_10k_cached_matches_cold_solo",
        ),
        (
            "src/serve/block_cache/tests/spill_model.rs",
            "restored_long_prefix_and_early_fork_match_cold_logits",
            "tests/nightly.sh",
            "serve::block_cache::tests::spill_model::restored_long_prefix_and_early_fork_match_cold_logits",
        ),
        (
            "src/serve/step_trace.rs",
            "dump",
            ".github/workflows/ci.yml",
            "serve::scheduler_trace::dump",
        ),
        (
            "src/spill/tests.rs",
            "kill9_mid_write_loses_cache_never_correctness",
            ".github/workflows/ci.yml",
            "spill::tests::kill9_mid_write_loses_cache_never_correctness",
        ),
        (
            "src/spill/tests.rs",
            "kill9_writer_helper",
            "src/spill/tests.rs",
            "spill::tests::kill9_writer_helper",
        ),
    ];

    let expected_keys: BTreeSet<(String, String)> = ledger
        .iter()
        .map(|(f, n, _, _)| ((*f).to_string(), (*n).to_string()))
        .collect();
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
