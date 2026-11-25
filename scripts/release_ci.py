"""Release gate helpers: wait for main-push CI and enforce immutable release tag invariants."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


REQUIRED_JOBS = {
    "unit",
    "go",
    "tla (memmode)",
    "tla (blocks)",
    "tla (grouped)",
    "tla (traces)",
    "m5-online-check",
    "m5-gpu",
    "m5-serve",
}

TAG_PATTERN = re.compile(r"^v([0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?)$")
SHA40_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def gh_json(*args):
    return json.loads(subprocess.check_output(["gh", *args], text=True))


def git_output(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True)


def git_check(*args, cwd=None):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def parse_cargo_version(cargo_toml_path):
    in_package = False
    for raw in Path(cargo_toml_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_package = line == "[package]"
            continue
        if in_package:
            m = re.match(r'^version\s*=\s*"([^"]+)"\s*$', line)
            if m:
                return m.group(1)
    raise ValueError(f"could not find [package] version in {cargo_toml_path}")


def parse_ls_remote_tag_sha(ls_remote_output, tag_name):
    direct_ref = f"refs/tags/{tag_name}"
    peeled_ref = f"refs/tags/{tag_name}^{{}}"
    direct_sha = None
    peeled_sha = None
    for raw in ls_remote_output.splitlines():
        parts = raw.strip().split()
        if len(parts) != 2:
            continue
        sha, ref = parts
        if not SHA40_PATTERN.match(sha):
            continue
        if ref == peeled_ref:
            peeled_sha = sha
        elif ref == direct_ref:
            direct_sha = sha
    resolved = peeled_sha or direct_sha
    if not resolved:
        raise ValueError(f"remote tag {direct_ref} was deleted or not found on origin")
    return resolved


def resolve_remote_tag_sha(remote, tag_name, cwd=None):
    out = git_output(
        "ls-remote",
        remote,
        f"refs/tags/{tag_name}",
        f"refs/tags/{tag_name}^{{}}",
        cwd=cwd,
    )
    return parse_ls_remote_tag_sha(out, tag_name)


def verify_remote_tag_unchanged(remote, tag_name, expected_sha, cwd=None):
    if not SHA40_PATTERN.match(expected_sha):
        raise ValueError(f"expected 40-hex commit SHA, got {expected_sha!r}")
    actual_sha = resolve_remote_tag_sha(remote, tag_name, cwd=cwd)
    if actual_sha != expected_sha:
        raise ValueError(
            f"remote tag refs/tags/{tag_name} moved during release: "
            f"expected {expected_sha}, found {actual_sha}"
        )
    return actual_sha


def verify_release_preflight(
    github_ref,
    ref_name,
    tag_sha,
    cargo_toml_path,
    notes_dir,
    crates_io_token,
    remote="origin",
    main_ref="origin/main",
    cwd=None,
    require_token=True,
):
    if not ref_name or not TAG_PATTERN.match(ref_name):
        raise ValueError(f"release ref_name must match vX.Y.Z, got {ref_name!r}")
    expected_ref = f"refs/tags/{ref_name}"
    if github_ref != expected_ref:
        raise ValueError(f"GITHUB_REF must be {expected_ref}, got {github_ref!r}")
    if not SHA40_PATTERN.match(tag_sha):
        raise ValueError(f"tag commit SHA must be 40 hex chars, got {tag_sha!r}")
    if require_token and not (crates_io_token or "").strip():
        raise ValueError(
            "CRATES_IO_TOKEN secret is not set or empty; refusing to build or publish a release"
        )

    cargo_version = parse_cargo_version(cargo_toml_path)
    tag_version = ref_name[1:]
    if cargo_version != tag_version:
        raise ValueError(
            f"tag {ref_name} ({tag_version}) does not match Cargo.toml version {cargo_version}"
        )

    notes_file = Path(notes_dir) / f"{ref_name}.md"
    if not notes_file.is_file() or not notes_file.read_text(encoding="utf-8").strip():
        raise ValueError(f"release notes {notes_file} are missing or empty")

    verify_remote_tag_unchanged(remote, ref_name, tag_sha, cwd=cwd)
    if not git_check("merge-base", "--is-ancestor", tag_sha, main_ref, cwd=cwd):
        raise ValueError(f"tagged commit {tag_sha} is not an ancestor of {main_ref}")


def require_main_push(run, sha):
    if (run["event"], run["headBranch"], run["headSha"]) != ("push", "main", sha):
        raise ValueError(f"CI run does not belong to the push of {sha} to main")


def completed_ci(repo, sha):
    runs = gh_json(
        "run", "list", "--repo", repo, "--workflow", "ci.yml", "--event", "push",
        "--branch", "main", "--commit", sha, "--limit", "1",
        "--json", "databaseId,headSha,headBranch,event,status",
    )
    if not runs:
        return None
    run = runs[0]
    require_main_push(run, sha)
    if run["status"] != "completed":
        return None
    # Read the current attempt, including jobs that a workflow-level success can skip.
    run = gh_json(
        "run", "view", str(run["databaseId"]), "--repo", repo,
        "--json", "headSha,headBranch,event,status,conclusion,jobs,url",
    )
    require_main_push(run, sha)
    if run["status"] != "completed":
        return None
    if run["conclusion"] != "success":
        raise ValueError(f'{run["url"]} concluded {run["conclusion"]}')
    for name in sorted(REQUIRED_JOBS):
        jobs = [job for job in run["jobs"] if job["name"] == name]
        if len(jobs) != 1 or (jobs[0]["status"], jobs[0]["conclusion"]) != ("completed", "success"):
            raise ValueError(f'{run["url"]} has no unique successful {name} job')
    return run["url"]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--verify-preflight"] or argv[:2] == ["--verify-preflight", "--no-token"]:
        require_token = "--no-token" not in argv
        root = Path.cwd()
        github_ref = os.environ.get("GITHUB_REF", "")
        ref_name = os.environ.get("GITHUB_REF_NAME", "")
        tag_sha = os.environ.get("TAG_SHA") or git_output("rev-parse", "HEAD").strip()
        token = os.environ.get("CARGO_REGISTRY_TOKEN") or os.environ.get("CRATES_IO_TOKEN", "")
        verify_release_preflight(
            github_ref=github_ref,
            ref_name=ref_name,
            tag_sha=tag_sha,
            cargo_toml_path=root / "Cargo.toml",
            notes_dir=root / "docs" / "releases",
            crates_io_token=token,
            remote="origin",
            main_ref="origin/main",
            cwd=root,
            require_token=require_token,
        )
        print(f"Release preflight verified: {github_ref} -> {tag_sha} (ancestor of origin/main)")
        return 0
    if argv[:1] == ["--verify-remote-tag"]:
        if len(argv) != 4:
            raise ValueError("usage: release_ci.py --verify-remote-tag <remote> <tag> <expected_sha>")
        _, remote, tag_name, expected_sha = argv
        verify_remote_tag_unchanged(remote, tag_name, expected_sha)
        print(f"Remote tag verified unchanged: refs/tags/{tag_name} -> {expected_sha}")
        return 0
    if len(argv) != 2:
        raise ValueError("usage: release_ci.py <repo> <sha>")
    repo, sha = argv
    print(f"Waiting for {', '.join(sorted(REQUIRED_JOBS))} on the main push of {sha}", flush=True)
    while True:
        url = completed_ci(repo, sha)
        if url:
            print(f"Release CI passed: {url}")
            return 0
        time.sleep(30)  # GitHub bounds the wait with the gate job's timeout.


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.CalledProcessError) as err:
        sys.exit(str(err))
