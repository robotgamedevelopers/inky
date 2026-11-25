"""Audit repository branch hygiene, already-integrated open PRs, and CI non-cancellation rules."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import release_ci


DEFAULT_PROTECTED_BRANCHES = frozenset({"main", "master", "HEAD"})


def load_workflow_yaml(path):
    """Parse a GitHub Actions YAML file using PyYAML or Ruby Psych."""
    try:
        import yaml
        return yaml.safe_load(Path(path).read_text())
    except ImportError:
        return json.loads(
            subprocess.check_output(
                ["ruby", "-ryaml", "-rjson", "-e", "puts JSON.generate(YAML.load_file(ARGV[0]))", str(path)],
                text=True,
            )
        )


def check_workflows_non_cancellation(workflows_dir):
    """Validate that every workflow file in workflows_dir refuses main-run cancellation."""
    wf_dir = Path(workflows_dir)
    if not wf_dir.is_dir():
        raise ValueError(f"workflows directory not found: {wf_dir}")
    paths = sorted(p for p in wf_dir.iterdir() if p.suffix in (".yml", ".yaml"))
    if not paths:
        raise ValueError(f"no workflow files found in {wf_dir}")
    checked = []
    for path in paths:
        doc = load_workflow_yaml(path)
        release_ci.validate_no_main_cancellation(doc, path.name)
        checked.append(path.name)
    return checked


def normalize_remote_branch(ref, remote="origin"):
    """Normalize a `git branch -r` entry to its branch name, or None for symbolic HEAD."""
    text = ref.strip()
    if not text or "->" in text:
        return None
    prefix = remote + "/"
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def find_merged_remote_branches(merged_refs, remote="origin", protected=DEFAULT_PROTECTED_BRANCHES):
    """Return sorted remote branch names already merged into base_ref, excluding protected refs."""
    out = set()
    for raw in merged_refs:
        name = normalize_remote_branch(raw, remote=remote)
        if name and name not in protected:
            out.add(name)
    return sorted(out)


def classify_open_prs(open_prs, ancestor_shas, merged_branches, cherry_integrated=None):
    """Classify open PRs into already-integrated, stale-base, or active-unmerged buckets."""
    merged_set = set(merged_branches)
    cherry_set = set(cherry_integrated or ())
    integrated_ancestors = []
    integrated_cherry = []
    stale_base_only = []
    active = []

    for pr in sorted(open_prs, key=lambda item: item["number"]):
        num = pr["number"]
        oid = pr["headRefOid"]
        base_ref = pr["baseRefName"]
        entry = {
            "number": num,
            "title": pr.get("title", ""),
            "headRefName": pr["headRefName"],
            "headRefOid": oid,
            "baseRefName": base_ref,
            "isDraft": bool(pr.get("isDraft", False)),
            "baseMerged": base_ref in merged_set,
        }
        if oid in ancestor_shas:
            entry["reason"] = "head commit is an ancestor of base ref"
            integrated_ancestors.append(entry)
        elif num in cherry_set:
            entry["reason"] = "all PR commits match patches already in base ref"
            integrated_cherry.append(entry)
        elif base_ref != "main" and base_ref in merged_set:
            entry["reason"] = f"target base branch {base_ref!r} is already merged into main"
            stale_base_only.append(entry)
        else:
            entry["reason"] = "unmerged changes remain"
            active.append(entry)

    return {
        "integrated_ancestors": integrated_ancestors,
        "integrated_cherry_picked": integrated_cherry,
        "stale_base": stale_base_only,
        "active": active,
    }


def build_hygiene_report(
    workflows_checked,
    merged_remote_branches,
    open_prs,
    ancestor_shas,
    cherry_integrated=None,
    remote="origin",
    base_ref="origin/main",
):
    """Assemble a structured hygiene audit report."""
    pr_buckets = classify_open_prs(
        open_prs=open_prs,
        ancestor_shas=ancestor_shas,
        merged_branches=merged_remote_branches,
        cherry_integrated=cherry_integrated,
    )
    stale_count = (
        len(merged_remote_branches)
        + len(pr_buckets["integrated_ancestors"])
        + len(pr_buckets["integrated_cherry_picked"])
    )
    return {
        "remote": remote,
        "base_ref": base_ref,
        "workflows_checked": list(workflows_checked),
        "merged_remote_branches": list(merged_remote_branches),
        "open_prs": pr_buckets,
        "stale_count": stale_count,
    }


def format_markdown_report(report):
    """Render a concise Markdown cleanup report without mutating repository state."""
    lines = [
        f"# Branch & PR Hygiene Report ({report['base_ref']})",
        "",
        f"- Workflows verified for non-cancellation: {', '.join(report['workflows_checked'])}",
        f"- Merged remote branches ready to prune ({len(report['merged_remote_branches'])}):",
    ]
    for branch in report["merged_remote_branches"]:
        lines.append(f"  - `{report['remote']}/{branch}`")

    ancestors = report["open_prs"]["integrated_ancestors"]
    cherry = report["open_prs"]["integrated_cherry_picked"]
    stale_base = report["open_prs"]["stale_base"]
    active = report["open_prs"]["active"]

    lines.append(f"- Already-integrated open PRs by commit ancestry ({len(ancestors)}):")
    for pr in ancestors:
        lines.append(
            f"  - `#{pr['number']}` `{pr['headRefName']}` -> `{pr['baseRefName']}` (`{pr['headRefOid'][:12]}`): {pr['reason']}"
        )

    lines.append(f"- Already-integrated open PRs by patch equivalence ({len(cherry)}):")
    for pr in cherry:
        lines.append(
            f"  - `#{pr['number']}` `{pr['headRefName']}` -> `{pr['baseRefName']}` (`{pr['headRefOid'][:12]}`): {pr['reason']}"
        )

    if stale_base:
        lines.append(f"- Open PRs targeting already-merged base branches ({len(stale_base)}):")
        for pr in stale_base:
            lines.append(
                f"  - `#{pr['number']}` `{pr['headRefName']}` -> `{pr['baseRefName']}`: {pr['reason']}"
            )

    lines.append(f"- Active / unmerged open PRs ({len(active)}):")
    for pr in active:
        draft_tag = " [draft]" if pr["isDraft"] else ""
        lines.append(
            f"  - `#{pr['number']}`{draft_tag} `{pr['headRefName']}` -> `{pr['baseRefName']}`"
        )
    return "\n".join(lines) + "\n"


def _git_lines(repo_root, *args):
    out = subprocess.check_output(["git", "-C", str(repo_root), *args], text=True)
    return [line for line in out.splitlines() if line.strip()]


def _is_ancestor(repo_root, commit, base_ref):
    res = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", commit, base_ref],
        capture_output=True,
    )
    return res.returncode == 0


def _all_commits_cherry_applied(repo_root, base_ref, head_ref_oid, base_branch, remote="origin"):
    """Return True if head_ref_oid has commits relative to remote/base_branch and every commit is '-' in git cherry."""
    target = f"{remote}/{base_branch}"
    check = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", target],
        capture_output=True,
    )
    if check.returncode != 0:
        return False
    lines = _git_lines(repo_root, "cherry", base_ref, head_ref_oid, target)
    if not lines:
        return False
    return all(line.startswith("- ") for line in lines)


def run_live_audit(repo_root, repo_slug, remote="origin", base_ref="origin/main"):
    """Run read-only git + gh hygiene audit against the repository."""
    root = Path(repo_root).resolve()
    workflows_checked = check_workflows_non_cancellation(root / ".github/workflows")
    merged_refs = _git_lines(root, "branch", "-r", "--merged", base_ref)
    merged_branches = find_merged_remote_branches(merged_refs, remote=remote)

    open_prs = json.loads(
        subprocess.check_output(
            [
                "gh", "pr", "list", "--repo", repo_slug, "--state", "open", "--limit", "100",
                "--json", "number,title,headRefName,headRefOid,baseRefName,isDraft",
            ],
            text=True,
        )
    )
    ancestor_shas = set()
    cherry_integrated = set()
    for pr in open_prs:
        oid = pr["headRefOid"]
        if _is_ancestor(root, oid, base_ref):
            ancestor_shas.add(oid)
        elif _all_commits_cherry_applied(root, base_ref, oid, pr["baseRefName"], remote=remote):
            cherry_integrated.add(pr["number"])

    return build_hygiene_report(
        workflows_checked=workflows_checked,
        merged_remote_branches=merged_branches,
        open_prs=open_prs,
        ancestor_shas=ancestor_shas,
        cherry_integrated=cherry_integrated,
        remote=remote,
        base_ref=base_ref,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Path to repository root")
    parser.add_argument("--repo", default="", help="GitHub owner/repo slug for live PR inspection")
    parser.add_argument("--remote", default="origin", help="Git remote name (default: origin)")
    parser.add_argument("--base-ref", default="origin/main", help="Base ref to check ancestry against")
    parser.add_argument("--offline-workflows", action="store_true", help="Only verify workflow non-cancellation rules offline")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown", help="Output format")
    parser.add_argument("--fail-on-stale", action="store_true", help="Exit 1 if merged branches or integrated PRs are found")
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    try:
        if args.offline_workflows:
            checked = check_workflows_non_cancellation(root / ".github/workflows")
            if args.format == "json":
                print(json.dumps({"workflows_checked": checked}, indent=2))
            else:
                print(f"Verified non-cancellation workflows: {', '.join(checked)}")
            return 0

        if not args.repo:
            parser.error("--repo is required unless --offline-workflows is set")

        report = run_live_audit(root, args.repo, remote=args.remote, base_ref=args.base_ref)
    except (ValueError, subprocess.CalledProcessError) as err:
        print(f"check_branch_hygiene: {err}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(format_markdown_report(report), end="")

    if args.fail_on_stale and report["stale_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
