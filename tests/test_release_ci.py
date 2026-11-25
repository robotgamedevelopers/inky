import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import release_ci


def workflow(path):
    # Read Actions YAML with the hosted runner's standard Ruby/Psych installation.
    return json.loads(subprocess.check_output(
        ["ruby", "-ryaml", "-rjson", "-e", "puts JSON.generate(YAML.load_file(ARGV[0]))", path],
        text=True,
    ))


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
        for key, value in {"--repo": self.repo, "--workflow": "ci.yml", "--event": "push",
                           "--branch": "main", "--commit": self.sha, "--limit": "1"}.items():
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

    def test_release_writes_require_the_gate(self):
        workflows = ROOT / ".github/workflows"
        jobs = workflow(workflows / "release.yml")["jobs"]
        gate = jobs["main-ci"]
        self.assertEqual(gate["permissions"], {"actions": "read", "contents": "read"})
        gate_steps = [step for step in gate["steps"] if "run" in step]
        self.assertIn('python3 scripts/release_ci.py "$GITHUB_REPOSITORY" "$(git rev-parse HEAD)"',
                      [step["run"] for step in gate_steps if not step.get("continue-on-error") and not step.get("if")])
        publisher = jobs["release"]
        self.assertEqual(publisher["needs"], "main-ci")
        self.assertNotIn("if", publisher)
        self.assertTrue(any("gh release create" in step.get("run", "") for step in publisher["steps"]))
        self.assertTrue(any("cargo publish" in step.get("run", "") for step in publisher["steps"]))

    def test_ci_runs_required_jobs_and_live_controls(self):
        ci_jobs = workflow(ROOT / ".github/workflows/ci.yml")["jobs"]
        expanded = set()
        for key, job in ci_jobs.items():
            if key in {"m5-approval", "m5-dispatch"}:
                self.assertIn("github.event_name == 'issue_comment'", job["if"])
                continue
            base = job.get("name", key)
            parts = job.get("strategy", {}).get("matrix", {}).get("part")
            if parts:
                expanded.update(f"{base} ({part})" for part in parts)
            else:
                expanded.add(base)
        self.assertEqual(expanded, release_ci.REQUIRED_JOBS)
        linux = ci_jobs["go"]["steps"]
        self.assertTrue(any("go test -race ./servecheck" in step.get("run", "")
                            and step.get("working-directory") == "tests" for step in linux))
        control = workflow(ROOT / "ci/inky-m5-control.yml")["jobs"]
        serve = [step.get("run", "") for step in control["m5-serve"]["steps"]
                 if not step.get("continue-on-error") and not step.get("if")]
        self.assertLess(
            serve.index("tests/serve.sh ./target/ci/inky"),
            serve.index("tests/serve-negative.sh tests/serve.sh ./target/ci/inky"),
        )


if __name__ == "__main__":
    unittest.main()
