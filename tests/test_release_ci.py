import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_m5_runner
import release_ci


def workflow(path):
    # Read Actions YAML with Ruby/Psych when installed, falling back to PyYAML on minimal Linux hosts.
    if shutil.which("ruby"):
        return json.loads(subprocess.check_output(
            ["ruby", "-ryaml", "-rjson", "-e", "puts JSON.generate(YAML.load_file(ARGV[0]))", path],
            text=True,
        ))
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


class ReleaseCiTests(unittest.TestCase):
    def setUp(self):
        self.repo = "owner/repo"
        self.sha = "a" * 40
        self.run = dict(databaseId=17, event="push", headBranch="main", headSha=self.sha,
                        status="completed", conclusion="success", url="https://example.test/run/17",
                        jobs=[dict(name=n, status="completed", conclusion="success")
                              for n in sorted(release_ci.REQUIRED_JOBS)])

    def check(self, listed=None, viewed=None):
        with patch.object(release_ci, "gh_json", side_effect=[
            [self.run] if listed is None else listed, self.run if viewed is None else viewed,
        ]) as gh:
            result = release_ci.completed_ci(self.repo, self.sha)
        args = gh.call_args_list[0].args
        for key, value in {
            "--repo": self.repo,
            "--workflow": "ci.yml",
            "--event": "push",
            "--branch": "main",
            "--commit": self.sha,
            "--limit": "1",
        }.items():
            self.assertEqual(args[args.index(key) + 1], value)
        return result

    def test_success_matches_exact_main_push_and_jobs(self):
        self.assertEqual(self.check(), self.run["url"])

    def test_uncreated_pending_and_rerun_in_progress_wait(self):
        self.assertIsNone(self.check(listed=[]))
        for status in ("queued", "in_progress"):
            self.assertIsNone(self.check(listed=[dict(self.run, status=status)]))
            self.assertIsNone(self.check(viewed=dict(self.run, status=status)))

    def test_wrong_identity_refused(self):
        for identity in ({"headSha": "b" * 40}, {"headBranch": "release"}, {"event": "pull_request"}):
            wrong = dict(self.run, **identity)
            with self.subTest(identity=identity):
                with self.assertRaises(ValueError):
                    self.check(listed=[wrong])
                with self.assertRaises(ValueError):
                    self.check(viewed=wrong)

    def test_failed_workflow_and_unqualified_required_jobs_refused(self):
        for conclusion in ("failure", "cancelled", "timed_out"):
            with self.assertRaises(ValueError):
                self.check(viewed=dict(self.run, conclusion=conclusion))
        for name in sorted(release_ci.REQUIRED_JOBS):
            other = [j for j in self.run["jobs"] if j["name"] != name]
            for bad in ([], [dict(name=name, status="completed", conclusion="skipped")],
                        [dict(name=name, status="in_progress", conclusion="")],
                        [dict(name=name, status="completed", conclusion="failure")]):
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    self.check(viewed=dict(self.run, jobs=other + bad))

    def test_release_preflight_and_remote_tag_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cargo = root / "Cargo.toml"
            cargo.write_text('[package]\nname = "inky-engine"\nversion = "0.2.4"\n', encoding="utf-8")
            notes = root / "notes"
            notes.mkdir()
            (notes / "v0.2.4.md").write_text("# inky v0.2.4\n", encoding="utf-8")

            tag_obj = "c" * 40
            peeled = self.sha
            annotated_ls = f"{tag_obj}\trefs/tags/v0.2.4\n{peeled}\trefs/tags/v0.2.4^{{}}\n"
            lightweight_ls = f"{peeled}\trefs/tags/v0.2.4\n"

            self.assertEqual(release_ci.parse_ls_remote_tag_sha(annotated_ls, "v0.2.4"), peeled)
            self.assertEqual(release_ci.parse_ls_remote_tag_sha(lightweight_ls, "v0.2.4"), peeled)
            with self.assertRaisesRegex(ValueError, "deleted or not found"):
                release_ci.parse_ls_remote_tag_sha("", "v0.2.4")

            with patch.object(release_ci, "git_output", return_value=annotated_ls), \
                 patch.object(release_ci, "git_check", return_value=True):
                release_ci.verify_release_preflight(
                    github_ref="refs/tags/v0.2.4",
                    ref_name="v0.2.4",
                    tag_sha=peeled,
                    cargo_toml_path=cargo,
                    notes_dir=notes,
                    crates_io_token="cio-secret-token",
                )

                # Empty CRATES_IO_TOKEN fails loudly up front (U132).
                for empty in ("", "   ", None):
                    with self.assertRaisesRegex(ValueError, "CRATES_IO_TOKEN"):
                        release_ci.verify_release_preflight(
                            github_ref="refs/tags/v0.2.4",
                            ref_name="v0.2.4",
                            tag_sha=peeled,
                            cargo_toml_path=cargo,
                            notes_dir=notes,
                            crates_io_token=empty,
                        )

                # Non-tag GITHUB_REF or mismatched ref_name rejected (U43, U133).
                with self.assertRaisesRegex(ValueError, "GITHUB_REF"):
                    release_ci.verify_release_preflight(
                        github_ref="refs/heads/main",
                        ref_name="v0.2.4",
                        tag_sha=peeled,
                        cargo_toml_path=cargo,
                        notes_dir=notes,
                        crates_io_token="cio-secret-token",
                    )

                # Cargo.toml version mismatch rejected.
                cargo.write_text('[package]\nname = "inky-engine"\nversion = "0.2.5"\n', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "does not match Cargo.toml"):
                    release_ci.verify_release_preflight(
                        github_ref="refs/tags/v0.2.4",
                        ref_name="v0.2.4",
                        tag_sha=peeled,
                        cargo_toml_path=cargo,
                        notes_dir=notes,
                        crates_io_token="cio-secret-token",
                    )
                cargo.write_text('[package]\nname = "inky-engine"\nversion = "0.2.4"\n', encoding="utf-8")

            # Not an ancestor of origin/main rejected (U43).
            with patch.object(release_ci, "git_output", return_value=annotated_ls), \
                 patch.object(release_ci, "git_check", return_value=False):
                with self.assertRaisesRegex(ValueError, "not an ancestor of origin/main"):
                    release_ci.verify_release_preflight(
                        github_ref="refs/tags/v0.2.4",
                        ref_name="v0.2.4",
                        tag_sha=peeled,
                        cargo_toml_path=cargo,
                        notes_dir=notes,
                        crates_io_token="cio-secret-token",
                    )

            # Force-moved or deleted tag during CI aborts before publish / gh release create (U43).
            moved_ls = f"{'d' * 40}\trefs/tags/v0.2.4\n"
            with patch.object(release_ci, "git_output", return_value=moved_ls):
                with self.assertRaisesRegex(ValueError, "moved during release"):
                    release_ci.verify_remote_tag_unchanged("origin", "v0.2.4", peeled)
            with patch.object(release_ci, "git_output", return_value=""):
                with self.assertRaisesRegex(ValueError, "deleted or not found"):
                    release_ci.verify_remote_tag_unchanged("origin", "v0.2.4", peeled)

    def test_release_writes_require_the_gate_and_tag_scoping(self):
        workflows = ROOT / ".github/workflows"
        jobs = workflow(workflows / "release.yml")["jobs"]
        gate = jobs["main-ci"]
        self.assertEqual(gate["permissions"], {"actions": "read", "contents": "read"})
        gate_steps = [step for step in gate["steps"] if "run" in step]
        self.assertIn('python3 scripts/release_ci.py "$GITHUB_REPOSITORY" "$(git rev-parse HEAD)"',
                      [step["run"] for step in gate_steps if not step.get("continue-on-error") and not step.get("if")])
        publisher = jobs["release"]
        self.assertEqual(publisher["needs"], "main-ci")
        self.assertEqual(publisher["environment"], "crates-io-release")
        self.assertNotIn("if", publisher)
        self.assertTrue(any("--verify-preflight" in step.get("run", "") for step in publisher["steps"]))
        release_steps = [step.get("run", "") for step in publisher["steps"] if "gh release create" in step.get("run", "")]
        publish_steps = [step.get("run", "") for step in publisher["steps"]
                         if "cargo publish --locked" in step.get("run", "") and "--dry-run" not in step.get("run", "")]
        self.assertEqual(len(release_steps), 1)
        self.assertEqual(len(publish_steps), 1)
        self.assertLess(release_steps[0].index("--verify-remote-tag"), release_steps[0].index("gh release create"))
        self.assertLess(publish_steps[0].index("--verify-remote-tag"), publish_steps[0].index("cargo publish --locked"))
        self.assertTrue((ROOT / "docs/RELEASE-TAG-PROTECTION.md").is_file())

    def test_ci_and_nightly_runner_checks_and_timeouts(self):
        # Pull requests and main pushes run unit tests and the smoke test only, on GitHub-hosted runners
        # (Glenn, 2026-11-25); everything else runs nightly on the fleet.
        ci_jobs = workflow(ROOT / ".github/workflows/ci.yml")["jobs"]
        self.assertEqual(set(ci_jobs), {"unit", "smoke"})
        self.assertEqual(set(ci_jobs), release_ci.REQUIRED_JOBS)
        for key, job in ci_jobs.items():
            self.assertIn("timeout-minutes", job, f"ci.yml job {key} missing timeout-minutes")
            self.assertIsInstance(job["runs-on"], str, f"ci.yml job {key} must run GitHub-hosted")
            self.assertFalse(job["runs-on"].startswith("self-hosted"), f"ci.yml job {key} runs self-hosted")
            self.assertNotIn("strategy", job, f"ci.yml job {key} must not fan out")
            for step in job["steps"]:
                run = step.get("run", "")
                self.assertNotIn("--ignored", run, f"ci.yml job {key} runs an ignored test: {run!r}")
                for slow in (r"cargo publish", r"soak", r"check-tla", r"\bgo (test|vet)\b", r"--release", r"tests/ci\.sh"):
                    self.assertIsNone(re.search(slow, run), f"ci.yml job {key} runs {slow!r}, which belongs nightly")
        unit = [step.get("run", "") for step in ci_jobs["unit"]["steps"]]
        for cmd in ("cargo clippy --locked --bins -- -D warnings", "cargo test --locked --bins --tests",
                    "python3 tests/test_branch_hygiene.py", "python3 tests/test_release_ci.py"):
            self.assertIn(cmd, unit)
        smoke = "\n".join(step.get("run", "") for step in ci_jobs["smoke"]["steps"])
        self.assertIn("cargo build --locked", smoke)
        self.assertIn("./target/debug/inky --help", smoke)

        linux = ["self-hosted", "Linux", "X64", "fleet", "tlc"]
        m5 = ["self-hosted", "macOS", "ARM64", "m5"]
        nightly_jobs = workflow(ROOT / ".github/workflows/nightly.yml")["jobs"]
        dep_jobs = workflow(ROOT / ".github/workflows/dependency-audit.yml")["jobs"]
        for wf_name, wf_jobs in (("nightly.yml", nightly_jobs), ("dependency-audit.yml", dep_jobs)):
            for key, job in wf_jobs.items():
                self.assertIn("timeout-minutes", job, f"{wf_name} job {key} missing timeout-minutes")
                self.assertIn(job["runs-on"], (linux, m5), f"{wf_name} job {key} is not on a fleet runner")
                self.assertFalse({"darwin", "services"} & set(job["runs-on"]), f"{wf_name} job {key} can land on a Studio")
                if key != "guard":
                    self.assertEqual(job.get("needs"), "guard", f"{wf_name} job {key} does not need guard")
                for step in job["steps"]:
                    self.assertNotIn("apt-get", step.get("run", ""), f"{wf_name} job {key} installs packages")
            self.assertIn("robotgamedevelopers/inky", wf_jobs["guard"]["steps"][0]["run"])
        expected = {"guard": linux, "go": linux, "tla": linux, "tla-scheduler": linux, "m5-online-check": linux,
                    "release-cpu": m5, "unit-extras": m5, "tla-layout-nightly": m5, "m5-full": m5}
        self.assertEqual({k: j["runs-on"] for k, j in nightly_jobs.items()}, expected)
        self.assertEqual(nightly_jobs["tla"]["strategy"]["matrix"]["part"], ["memmode", "blocks", "grouped", "traces"])
        go = nightly_jobs["go"]["steps"]
        self.assertTrue(any("go test -race ./servecheck" in step.get("run", "")
                            and step.get("working-directory") == "tests" for step in go))
        self.assertTrue(any("command -v bwrap" in step.get("run", "") for step in go))
        extras = "\n".join(step.get("run", "") for step in nightly_jobs["unit-extras"]["steps"])
        for needle in ("tests/soak_qualification_test.sh", "check_branch_hygiene.py --offline-workflows",
                       "serve_header_policy", "memd::tests:: -- --ignored", "kill9_mid_write_loses_cache_never_correctness",
                       "serve::scheduler_trace::dump"):
            self.assertIn(needle, extras)
        full = [step for step in nightly_jobs["m5-full"]["steps"] if "run" in step]
        for step in full[2:]:
            self.assertEqual(step.get("if"), "${{ !cancelled() }}", f"m5-full step {step.get('name')} can be hidden")
        full_runs = "\n".join(step["run"] for step in full)
        for gate in ("router_handles_nonfinite_scores", "k_attn_b_and_blk_are_split_invariant_across_sliding_window",
                     "gpu_pool_zero_fills_a_taken_slot", "dspark_gpu_guards_and_rejection_sampling_gate",
                     "engine_encode_decode_64_positions_syncs_once_then_63_zero_steps",
                     "real_gguf_tokenizer_74k_corpus_roundtrip_and_determinism",
                     "gguf_embedded_chat_template_matches_serve_prompt_render_token_for_token",
                     "tests/ci.sh full", "tests/serve.sh", "tests/serve-negative.sh tests/serve.sh"):
            self.assertIn(gate, full_runs)
        self.assertNotIn("--profile ci", full_runs)
        chk_steps = " ".join(s.get("run", "") for s in nightly_jobs["m5-online-check"]["steps"])
        self.assertNotIn("--cancel-on-offline", chk_steps)
        self.assertIn("--jobs m5-full,release-cpu,unit-extras,tla-layout-nightly", chk_steps)
        self.assertEqual(nightly_jobs["m5-online-check"]["permissions"], {"actions": "read", "contents": "read"})

        dep_doc = workflow(ROOT / ".github/workflows/dependency-audit.yml")
        dep_on = dep_doc["on"] if "on" in dep_doc else dep_doc["true"]  # Psych reads the bare key on as true.
        self.assertEqual(set(dep_on), {"schedule", "workflow_dispatch"})
        self.assertEqual(dep_on["schedule"], [{"cron": "17 6 * * *"}])

    def test_workflows_pin_actions_by_sha_rust_toolchain_and_locked_cargo(self):
        import re
        sha_pin = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}$")
        workflows = ROOT / ".github/workflows"
        for wf_name in ("ci.yml", "nightly.yml", "dependency-audit.yml", "release.yml"):
            doc = workflow(workflows / wf_name)
            for job_name, job in doc["jobs"].items():
                steps = job.get("steps", [])
                saw_rust_pin = False
                runs_cargo = False
                for step in steps:
                    if "uses" in step:
                        uses = step["uses"]
                        self.assertRegex(
                            uses, sha_pin,
                            f"{wf_name}:{job_name} uses unpinned action {uses!r}",
                        )
                        if uses.startswith("dtolnay/rust-toolchain@"):
                            self.assertEqual(str(step.get("with", {}).get("toolchain")), "1.98.1")
                            saw_rust_pin = True
                    run_cmd = step.get("run", "")
                    if "rustc 1.98.1" in run_cmd:
                        saw_rust_pin = True
                    for line in run_cmd.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("#") or "cargo " not in stripped or stripped.startswith("cargo audit "):
                            continue
                        runs_cargo = True
                        self.assertIn(
                            "--locked", stripped,
                            f"{wf_name}:{job_name} runs cargo without --locked: {stripped!r}",
                        )
                if runs_cargo:
                    self.assertTrue(
                        saw_rust_pin,
                        f"{wf_name}:{job_name} runs cargo without pinning Rust 1.98.1",
                    )

    def test_linux_fleet_jobs_are_not_m5_activity(self):
        fleet = {"name": "go", "status": "in_progress", "labels": ["self-hosted", "Linux", "X64", "fleet", "tlc"],
                 "runner_name": "hetzner1"}
        self.assertFalse(check_m5_runner.is_m5_job(fleet, {"m5-full"}))
        self.assertFalse(check_m5_runner.is_recently_active_m5_job(fleet, {"m5-full"}, 0.0))
        m5 = dict(fleet, name="release-cpu", labels=["self-hosted", "macOS", "ARM64", "m5"])
        self.assertTrue(check_m5_runner.is_m5_job(m5, {"m5-full"}))

    def test_m5_runner_availability_direct_and_github_token_fallback(self):
        m5_labels = [{"name": "self-hosted"}, {"name": "macOS"}, {"name": "ARM64"}, {"name": "m5"}]

        # 1. Direct /actions/runners 200 with online M5 runner -> immediate 0.
        out = io.StringIO()
        rc = check_m5_runner.check_m5_availability(
            "robotgamedevelopers/inky", 100, ["m5-full", "unit-extras"],
            api_get=lambda ep: {"runners": [{"name": "m5-ci-runner-1", "status": "online", "labels": m5_labels}]},
            out=out,
        )
        self.assertEqual(rc, 0)
        self.assertIn("m5-ci-runner-1", out.getvalue())

        # 2. Direct /actions/runners 200 with all M5 runners offline -> immediate ::error:: + 1 (no cancel by default).
        out = io.StringIO()
        cancelled = []
        rc = check_m5_runner.check_m5_availability(
            "robotgamedevelopers/inky", 100, ["m5-full"],
            api_get=lambda ep: {"runners": [{"name": "m5-ci-runner-1", "status": "offline", "labels": m5_labels}]},
            api_post=cancelled.append,
            out=out,
        )
        self.assertEqual(rc, 1)
        self.assertIn("::error::No online M5 self-hosted runner", out.getvalue())
        self.assertEqual(cancelled, [])

        # 3. GITHUB_TOKEN 403 on /actions/runners + our run's M5 job transitions to in_progress -> 0.
        calls = {"own": 0}
        def api_picked_up(ep):
            if ep.endswith("/actions/runners?per_page=100"):
                raise check_m5_runner.GhHttpError(403, "HTTP 403 Resource not accessible by integration")
            if "/actions/runs/100/jobs" in ep:
                calls["own"] += 1
                st = "queued" if calls["own"] == 1 else "in_progress"
                return {"jobs": [{"name": "m5-full", "status": st}]}
            if "status=in_progress" in ep or ep.endswith("/actions/runs?per_page=25"):
                return {"workflow_runs": [{"id": 100}]}
            raise AssertionError(ep)

        now = [0.0]
        rc = check_m5_runner.check_m5_availability(
            "robotgamedevelopers/inky", 100, ["m5-full", "unit-extras"],
            max_queue_seconds=180, poll_seconds=5,
            api_get=api_picked_up,
            clock=lambda: now[0],
            sleep_fn=lambda d: now.__setitem__(0, now[0] + d),
            out=io.StringIO(),
        )
        self.assertEqual(rc, 0)
        self.assertEqual(now[0], 5.0)

        # 4. GITHUB_TOKEN 403 + our run queued while another queued run has a recently completed M5 job -> 0.
        def api_recent_other(ep):
            if ep.endswith("/actions/runners?per_page=100"):
                raise check_m5_runner.GhHttpError(403, "HTTP 403")
            if "/actions/runs/100/jobs" in ep:
                return {"jobs": [{"name": "m5-full", "status": "queued"}]}
            if "status=in_progress" in ep:
                return {"workflow_runs": []}
            if ep.endswith("/actions/runs?per_page=25"):
                return {"workflow_runs": [{"id": 99, "status": "queued", "updated_at": "2026-11-25T12:40:00Z"}]}
            if "/actions/runs/99/jobs" in ep:
                return {"jobs": [{
                    "name": "m5",
                    "status": "completed",
                    "conclusion": "success",
                    "runner_name": "mini-m5-inky-1",
                    "completed_at": "2026-11-25T12:39:30Z",
                }]}
            raise AssertionError(ep)

        rc = check_m5_runner.check_m5_availability(
            "robotgamedevelopers/inky", 100, ["m5-full"],
            api_get=api_recent_other,
            clock=lambda: 0.0,
            wall_time=lambda: check_m5_runner.parse_iso_epoch("2026-11-25T12:42:00Z"),
            sleep_fn=lambda _: (_ for _ in ()).throw(AssertionError("should not sleep when recently active")),
            out=io.StringIO(),
        )
        self.assertEqual(rc, 0)

        # 5. GITHUB_TOKEN 403 + queued > 180s with 0 active M5 runs across repo -> ::error:: + exit 1.
        def api_all_offline(ep):
            if ep.endswith("/actions/runners?per_page=100"):
                raise check_m5_runner.GhHttpError(403, "HTTP 403")
            if "/actions/runs/100/jobs" in ep:
                return {"jobs": [{"name": "m5-full", "status": "queued"}, {"name": "unit-extras", "status": "queued"}]}
            if "status=in_progress" in ep or ep.endswith("/actions/runs?per_page=25"):
                return {"workflow_runs": [{"id": 100}]}
            raise AssertionError(ep)

        now = [0.0]
        cancelled = []
        out = io.StringIO()
        rc = check_m5_runner.check_m5_availability(
            "robotgamedevelopers/inky", 100, ["m5-full", "unit-extras"],
            max_queue_seconds=180, poll_seconds=15, cancel_on_offline=True,
            api_get=api_all_offline,
            api_post=cancelled.append,
            clock=lambda: now[0],
            sleep_fn=lambda d: now.__setitem__(0, now[0] + d),
            out=out,
        )
        self.assertEqual(rc, 1)
        self.assertEqual(now[0], 180.0)
        self.assertIn("::error::No online M5 self-hosted runner picked up the job (0 active M5 runs", out.getvalue())
        self.assertEqual(cancelled, ["repos/robotgamedevelopers/inky/actions/runs/100/cancel"])


    def test_workflows_never_cancel_main_runs(self):
        workflows_dir = ROOT / ".github/workflows"
        wf_paths = sorted(
            p for p in workflows_dir.iterdir() if p.suffix in (".yml", ".yaml")
        )
        self.assertGreaterEqual(len(wf_paths), 4)
        for path in wf_paths:
            with self.subTest(workflow=path.name):
                doc = workflow(path)
                release_ci.validate_no_main_cancellation(doc, path.name)

    def test_planted_main_cancellation_bugs_fail(self):
        ci_doc = workflow(ROOT / ".github/workflows/ci.yml")
        nightly_doc = workflow(ROOT / ".github/workflows/nightly.yml")
        dep_doc = workflow(ROOT / ".github/workflows/dependency-audit.yml")
        release_doc = workflow(ROOT / ".github/workflows/release.yml")

        cases = [
            (
                "ci.yml cancel-in-progress true on main",
                dict(ci_doc, concurrency={"group": "ci-main", "cancel-in-progress": True}),
            ),
            (
                "ci.yml missing concurrency block",
                {k: v for k, v in ci_doc.items() if k != "concurrency"},
            ),
            (
                "ci.yml string 'true' cancel-in-progress",
                dict(ci_doc, concurrency={"group": "ci-${{ github.ref }}", "cancel-in-progress": "true"}),
            ),
            (
                "ci.yml PR cancel expression without run_id fallback group",
                dict(
                    ci_doc,
                    concurrency={
                        "group": "ci-${{ github.ref }}",
                        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
                    },
                ),
            ),
            (
                "nightly.yml cancel-in-progress true",
                dict(nightly_doc, concurrency={"group": "nightly-${{ github.ref }}", "cancel-in-progress": True}),
            ),
            (
                "dependency-audit.yml non-PR-restricted expression",
                dict(
                    dep_doc,
                    concurrency={
                        "group": "audit-${{ github.ref }}",
                        "cancel-in-progress": "${{ github.ref == 'refs/heads/main' }}",
                    },
                ),
            ),
            (
                "release.yml cancel-in-progress true",
                dict(release_doc, concurrency={"group": "release-${{ github.ref }}", "cancel-in-progress": True}),
            ),
            (
                "job-level cancel-in-progress true on main",
                dict(
                    ci_doc,
                    jobs={
                        **ci_doc["jobs"],
                        "unit": {
                            **ci_doc["jobs"]["unit"],
                            "concurrency": {"group": "unit-main", "cancel-in-progress": True},
                        },
                    },
                ),
            ),
        ]
        for label, bad_doc in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    release_ci.validate_no_main_cancellation(bad_doc, label)


if __name__ == "__main__":
    unittest.main()
