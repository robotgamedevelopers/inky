import copy
import sys
import unittest

from test_release_ci import ROOT, workflow

sys.path.insert(0, str(ROOT / "scripts"))
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

    def test_live_workflows_use_restricted_group_and_isolate_the_publisher(self):
        files = ROOT / ".github/workflows"
        ci = workflow(files / "ci.yml")["jobs"]
        nightly = workflow(files / "nightly.yml")["jobs"]
        for key in ("m5-gpu", "m5-serve"):
            job = ci[key]
            self.assertEqual(job["runs-on"]["group"], "inky-reviewed-m5")
            self.assertIn("github.ref == 'refs/heads/main'", job["if"])
            self.assertIn("needs.m5-approval.result == 'success'", job["if"])
            self.assertNotIn("github.event_name == 'pull_request'", job["if"])
            checkout = job["steps"][0]["with"]
            self.assertEqual(checkout["ref"], "${{ needs.m5-approval.outputs.candidate || github.sha }}")
            self.assertFalse(checkout["persist-credentials"])
        self.assertEqual(nightly["m5-full"]["runs-on"]["group"], "inky-reviewed-m5")
        self.assertIn("github.ref == 'refs/heads/main'", nightly["m5-full"]["if"])
        approval, publisher = ci["m5-approval"], ci["m5-attestation"]
        self.assertEqual(approval["runs-on"], "ubuntu-latest")
        self.assertEqual(publisher["runs-on"], "ubuntu-latest")
        self.assertTrue(all("checkout" not in step.get("uses", "") for step in publisher["steps"]))
        self.assertEqual(publisher["environment"], "inky-m5-attestation")
        self.assertEqual(publisher["permissions"], {})
        self.assertEqual(set(publisher["needs"]), {"m5-approval", "m5-gpu", "m5-serve"})
        step = publisher["steps"][-1]
        self.assertEqual(step["env"]["HEAD_SHA"], "${{ needs.m5-approval.outputs.head }}")
        self.assertIn('-f head_sha="$HEAD_SHA"', step["run"])
        self.assertIn('[[ "$M5_GPU" = success && "$M5_SERVE" = success ]]', step["run"])


if __name__ == "__main__":
    unittest.main()
