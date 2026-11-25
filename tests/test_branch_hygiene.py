import io
import json
from pathlib import Path
import sys
from contextlib import redirect_stdout
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_branch_hygiene


class BranchHygieneTests(unittest.TestCase):
    def test_find_merged_remote_branches_excludes_protected_and_symbolic_head(self):
        refs = [
            "  origin/HEAD -> origin/main",
            "  origin/main",
            "  origin/master",
            "  origin/stella/nov25-sampling",
            "  origin/emma/release-v0.2.9",
            "  origin/stella/nov25-sampling",
        ]
        got = check_branch_hygiene.find_merged_remote_branches(refs)
        self.assertEqual(got, ["emma/release-v0.2.9", "stella/nov25-sampling"])

    def test_classify_open_prs_separates_integrated_stale_and_active(self):
        prs = [
            {
                "number": 86,
                "title": "Validate GPU samples",
                "headRefName": "stella/nov25-sampling",
                "headRefOid": "51e397345f05e4d2d4aa823d6adc41e2f1a9d4ee",
                "baseRefName": "stella/nov25-engine-runtime",
                "isDraft": False,
            },
            {
                "number": 89,
                "title": "Count TLA coverage only for scheduled runner steps",
                "headRefName": "stella/nov25-ci51",
                "headRefOid": "e2a93de2694c186ae65fad52b0acabb8cb9ad257",
                "baseRefName": "rowan/ci-one-minute",
                "isDraft": False,
            },
            {
                "number": 104,
                "title": "Split modules",
                "headRefName": "stella/nov25-module-splits",
                "headRefOid": "09bf8ead0f6cb1bb6594cab5b0bf03f8e2e59b9d",
                "baseRefName": "stella/nov25-integration-main",
                "isDraft": False,
            },
            {
                "number": 126,
                "title": "Bump setup-go",
                "headRefName": "dependabot/github_actions/actions/setup-go-7.1.0",
                "headRefOid": "3d3dfcf517926344824b68b1943daa19a1f8a82f",
                "baseRefName": "main",
                "isDraft": False,
            },
        ]
        ancestors = {
            "51e397345f05e4d2d4aa823d6adc41e2f1a9d4ee",
            "09bf8ead0f6cb1bb6594cab5b0bf03f8e2e59b9d",
        }
        merged_branches = ["stella/nov25-engine-runtime", "stella/nov25-integration-main"]
        report = check_branch_hygiene.build_hygiene_report(
            workflows_checked=["ci.yml", "dependency-audit.yml", "nightly.yml", "release.yml"],
            merged_remote_branches=merged_branches,
            open_prs=prs,
            ancestor_shas=ancestors,
            cherry_integrated={89},
        )
        self.assertEqual([p["number"] for p in report["open_prs"]["integrated_ancestors"]], [86, 104])
        self.assertEqual([p["number"] for p in report["open_prs"]["integrated_cherry_picked"]], [89])
        self.assertEqual([p["number"] for p in report["open_prs"]["active"]], [126])
        self.assertEqual(report["stale_count"], 5)

        md = check_branch_hygiene.format_markdown_report(report)
        self.assertIn("#86", md)
        self.assertIn("#89", md)
        self.assertIn("#104", md)
        self.assertIn("origin/stella/nov25-engine-runtime", md)

    def test_offline_workflows_check_passes_repo_workflows(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = check_branch_hygiene.main(
                ["--repo-root", str(ROOT), "--offline-workflows", "--format", "json"]
            )
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(
            payload["workflows_checked"],
            ["ci.yml", "dependency-audit.yml", "nightly.yml", "release.yml"],
        )

    def test_planted_stale_branch_or_cancelled_workflow_fails(self):
        # 1. Missing workflows directory must fail non-zero.
        self.assertEqual(
            check_branch_hygiene.main(
                ["--repo-root", str(ROOT / "nonexistent-dir"), "--offline-workflows"]
            ),
            1,
        )
        # 2. Stale integrated PR / merged branch triggers fail-on-stale via report structure.
        report = check_branch_hygiene.build_hygiene_report(
            workflows_checked=["ci.yml"],
            merged_remote_branches=["old-merged-branch"],
            open_prs=[],
            ancestor_shas=set(),
        )
        self.assertGreater(report["stale_count"], 0)


if __name__ == "__main__":
    unittest.main()
