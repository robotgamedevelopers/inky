import copy
import sys
import unittest

from test_release_ci import ROOT, workflow

sys.path.insert(0, str(ROOT / "scripts"))
from reviewed_m5 import REPOSITORY, resolve_candidate


HEAD, BASE, MERGE = "a" * 40, "b" * 40, "c" * 40


class ReviewedM5Tests(unittest.TestCase):
    def setUp(self):
        self.event = {
            "comment": {"body": f"/test-m5 {HEAD}", "user": {"login": "reviewer", "type": "User"}},
            "issue": {"number": 41, "pull_request": {}},
        }
        self.event["issue"]["pull_request"]["url"] = "https://api.github.com/pulls/41"
        self.pull = {
            "state": "open", "draft": False, "base": {"repo": {"full_name": REPOSITORY}, "ref": "main"},
            "head": {"sha": HEAD}, "user": {"login": "author"},
            "merge_commit_sha": MERGE, "mergeable": True,
        }
        self.approval = {"user": {"login": "reviewer"}, "state": "APPROVED", "commit_id": HEAD}
        self.path = "pulls/41"
        self.responses = {
            "collaborators/reviewer/permission": {"permission": "write"},
            self.path: self.pull,
            f"{self.path}/reviews?per_page=100": [[self.approval]],
            "git/ref/heads/main": {"object": {"sha": BASE}},
            f"git/commits/{MERGE}": {"parents": [{"sha": BASE}, {"sha": HEAD}]},
        }

    def request(self, path, *args):
        if path.endswith("/reviews?per_page=100"):
            self.assertEqual(args, ("--paginate", "--slurp"))
        return copy.deepcopy(self.responses[path])

    def resolve(self):
        return resolve_candidate(self.event, self.request)

    def test_approved_exact_merge_candidate(self):
        self.assertEqual(self.resolve(), MERGE)

    def test_changed_head_or_base_refused(self):
        self.pull["head"]["sha"] = "d" * 40
        with self.assertRaisesRegex(ValueError, "current pull request head"):
            self.resolve()
        self.pull["head"]["sha"] = HEAD
        self.responses["git/ref/heads/main"]["object"]["sha"] = "e" * 40
        with self.assertRaisesRegex(ValueError, "current main"):
            self.resolve()

    def test_missing_foreign_or_stale_approval_refused(self):
        key = f"{self.path}/reviews?per_page=100"
        for pages in ([], [[dict(self.approval, commit_id="e" * 40)]],
                      [[dict(self.approval, user={"login": "other"})]],
                      [[self.approval], [dict(self.approval, state="CHANGES_REQUESTED")]],
                      [[dict(self.approval, state="DISMISSED")]]):
            with self.subTest(reviews=pages), self.assertRaisesRegex(ValueError, "fresh approving review"):
                self.responses[key] = pages
                self.resolve()

    def test_read_only_reviewer_and_self_approval_refused(self):
        permission = self.responses["collaborators/reviewer/permission"]
        permission["permission"] = "read"
        with self.assertRaisesRegex(ValueError, "write access"):
            self.resolve()
        permission["permission"] = "write"
        self.pull["user"]["login"] = "Reviewer"
        with self.assertRaisesRegex(ValueError, "author cannot"):
            self.resolve()

    def test_malformed_command_refused_without_api_call(self):
        self.event["comment"]["body"] = "/test-m5 refs/heads/feature"
        with self.assertRaisesRegex(ValueError, "full approved PR head SHA"):
            resolve_candidate(self.event, lambda *_: self.fail("unexpected API call"))

    def test_racing_update_refused(self):
        def request(path, *args):
            response = self.request(path, *args)
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


if __name__ == "__main__":
    unittest.main()
