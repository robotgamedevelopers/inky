import copy
import sys
import unittest
from unittest.mock import patch

from test_release_ci import ROOT, workflow

sys.path.insert(0, str(ROOT / "scripts"))
from m5_control import dispatch, receipt
from reviewed_m5 import REPOSITORY, resolve_candidate


HEAD, BASE, MERGE, TREE = "a" * 40, "b" * 40, "c" * 40, "d" * 40


class ReviewedM5Tests(unittest.TestCase):
    def setUp(self):
        self.event = {
            "comment": {"body": f"/test-m5 {HEAD}", "user": {"login": "reviewer", "type": "User"}},
            "issue": {"number": 41, "pull_request": {"url": "https://api.github.com/pulls/41"}},
        }
        self.pull = {
            "state": "open", "draft": False, "base": {"repo": {"full_name": REPOSITORY}, "ref": "main"},
            "head": {"sha": HEAD}, "user": {"login": "author"},
            "merge_commit_sha": MERGE, "mergeable": True,
        }
        self.path = "pulls/41"
        self.responses = {
            "collaborators/reviewer/permission": {"permission": "admin"},
            self.path: self.pull,
            "git/ref/heads/main": {"object": {"sha": BASE}},
            f"compare/{BASE}...{HEAD}?per_page=1": {"behind_by": 0, "merge_base_commit": {"sha": BASE}},
            f"git/commits/{MERGE}": {"parents": [{"sha": BASE}, {"sha": HEAD}], "tree": {"sha": TREE}},
            f"git/commits/{HEAD}": {"tree": {"sha": TREE}},
        }

    def request(self, path):
        return copy.deepcopy(self.responses[path])

    def resolve(self):
        return resolve_candidate(self.event, self.request)

    def test_approved_exact_merge_candidate(self):
        self.assertEqual(self.resolve(), (HEAD, MERGE))

    def test_changed_head_or_base_refused(self):
        self.pull["head"]["sha"] = "d" * 40
        with self.assertRaisesRegex(ValueError, "current pull request head"):
            self.resolve()
        self.pull["head"]["sha"] = HEAD
        self.responses[f"git/commits/{MERGE}"]["parents"][0]["sha"] = "e" * 40
        with self.assertRaisesRegex(ValueError, "current main"):
            self.resolve()

    def test_stale_branch_and_mismatched_tree_refused(self):
        relation = self.responses[f"compare/{BASE}...{HEAD}?per_page=1"]
        relation["behind_by"] = 1
        with self.assertRaisesRegex(ValueError, "include current main"):
            self.resolve()
        relation["behind_by"] = 0
        self.responses[f"git/commits/{HEAD}"]["tree"]["sha"] = "e" * 40
        with self.assertRaisesRegex(ValueError, "different trees"):
            self.resolve()

    def test_same_account_admin_allowed_non_admin_and_bot_refused(self):
        self.pull["user"]["login"] = "Reviewer"
        self.assertEqual(self.resolve(), (HEAD, MERGE))
        permission = self.responses["collaborators/reviewer/permission"]
        for role in ("read", "write", "maintain", "none"):
            with self.subTest(role=role):
                permission["permission"] = role
                with self.assertRaisesRegex(ValueError, "administrator access"):
                    self.resolve()
        permission["permission"] = "admin"
        self.event["comment"]["user"]["type"] = "Bot"
        with self.assertRaisesRegex(ValueError, "human reviewer"):
            self.resolve()

    def test_malformed_command_refused_without_api_call(self):
        self.event["comment"]["body"] = "/test-m5 refs/heads/feature"
        with self.assertRaisesRegex(ValueError, "full approved PR head SHA"):
            resolve_candidate(self.event, lambda *_: self.fail("unexpected API call"))

    def test_racing_update_refused(self):
        def request(path):
            response = self.request(path)
            if path.startswith("git/commits/"):
                self.pull["head"]["sha"] = "f" * 40
            return response

        with self.assertRaisesRegex(ValueError, "changed while resolving"):
            resolve_candidate(self.event, request)

    def test_dispatch_targets_private_main_and_binds_attempt(self):
        environment = {"M5_CONTROL_REPOSITORY": "inky-ci", "GITHUB_RUN_ID": "123",
                       "GITHUB_RUN_ATTEMPT": "2"}
        with patch.dict("os.environ", environment), patch("m5_control.api") as api, patch("sys.stdout"):
            dispatch("pr", HEAD, MERGE)
            api.assert_called_once_with(
                "repos/robotgamedevelopers/inky-ci/actions/workflows/inky-m5.yml/dispatches",
                "-f", "ref=main", "-f", "inputs[mode]=pr", "-f", f"inputs[head]={HEAD}",
                "-f", f"inputs[candidate]={MERGE}", "-f", "inputs[request]=123.2",
            )
        with patch.dict("os.environ", dict(environment, M5_CONTROL_REPOSITORY="inky")), patch("m5_control.api") as api:
            with self.assertRaisesRegex(ValueError, "separate private"):
                dispatch("pr", HEAD, MERGE)
            api.assert_not_called()

    def test_receipt_matches_reporter_commit_and_exact_attempt(self):
        valid = {"name": "m5-main-gpu", "head_sha": HEAD, "external_id": "123.2",
                 "app": {"id": 42}, "status": "completed", "conclusion": "success", "id": 7}
        for fields in ({"app": {"id": 15368}}, {"head_sha": BASE}, {"external_id": "123.1"},
                       {"status": "queued"}, {"name": "m5-main-serve"}, {"app": None}):
            check = dict(valid, **fields)
            with self.subTest(fields=fields):
                self.assertIsNone(receipt([{"check_runs": [check]}], 42, HEAD, "m5-main-gpu", "123.2"))
        newer = dict(valid, id=9, conclusion="failure")
        pages = [{"check_runs": [valid]}, {"check_runs": [newer]}]
        self.assertEqual(receipt(pages, 42, HEAD, "m5-main-gpu", "123.2"), newer)

    def test_live_workflows_separate_dispatch_tests_and_publisher(self):
        files = ROOT / ".github/workflows"
        ci = workflow(files / "ci.yml")["jobs"]
        nightly = workflow(files / "nightly.yml")["jobs"]
        control = workflow(ROOT / "ci/inky-m5-control.yml")["jobs"]
        for public in (ci, nightly):
            self.assertTrue(all(isinstance(job["runs-on"], str) for job in public.values()))
        dispatch = ci["m5-dispatch"]
        self.assertEqual(dispatch["environment"], "inky-m5-dispatch")
        self.assertIn("github.ref == 'refs/heads/main'", dispatch["if"])
        self.assertIn("needs.m5-approval.result == 'success'", dispatch["if"])
        self.assertNotIn("github.event_name == 'pull_request'", dispatch["if"])
        self.assertEqual(nightly["m5-full"]["environment"], dispatch["environment"])
        for name in ("m5-gpu", "m5-serve", "m5-full"):
            job = control[name]
            self.assertIn("self-hosted", job["runs-on"])
            self.assertEqual(job["needs"], "input")
            self.assertNotIn("environment", job)
            checkout = job["steps"][0]
            self.assertIn('git fetch --no-tags --depth=1 origin "$CANDIDATE"', checkout["run"])
            self.assertIn('test "$(git rev-parse HEAD)" = "$CANDIDATE"', checkout["run"])
        approval = control["input"]
        self.assertEqual(approval["runs-on"], "ubuntu-latest")
        self.assertIn('"$ACTOR" = "$DISPATCH_ACTOR"', approval["steps"][0]["run"])
        publisher = control["attest"]
        self.assertEqual(publisher["runs-on"], "ubuntu-latest")
        self.assertTrue(all("checkout" not in step.get("uses", "") for step in publisher["steps"]))
        self.assertEqual(publisher["environment"], "inky-m5-report")
        self.assertEqual(set(publisher["needs"]), {"input", "m5-gpu", "m5-serve", "m5-full"})
        step = publisher["steps"][-1]
        self.assertIn('-f head_sha="$HEAD" -f external_id="$REQUEST"', step["run"])
        self.assertIn('[[ "$GPU" = success && "$SERVE" = success ]]', step["run"])


if __name__ == "__main__":
    unittest.main()
