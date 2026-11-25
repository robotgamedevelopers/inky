"""Wait for the tagged commit's main-push CI before publishing any release assets."""
import json
import re
import subprocess
import sys
import time


_PR_ONLY_CANCEL_RE = re.compile(
    r"^\$\{\{\s*github\.event_name\s*==\s*(['\"])pull_request\1\s*\}\}$"
)


def _validate_concurrency_block(block, location):
    if not isinstance(block, dict):
        raise ValueError(f"{location}: missing or non-mapping concurrency block")
    group = block.get("group")
    if not isinstance(group, str) or not group.strip():
        raise ValueError(f"{location}: concurrency.group must be a non-empty string")
    if "cancel-in-progress" not in block:
        raise ValueError(f"{location}: concurrency must set cancel-in-progress")
    cancel = block["cancel-in-progress"]
    if cancel is False:
        return
    if isinstance(cancel, str) and _PR_ONLY_CANCEL_RE.match(cancel.strip()):
        if "github.run_id" not in group:
            raise ValueError(
                f"{location}: pull_request cancel-in-progress group {group!r} must fall back to github.run_id for non-PR runs"
            )
        return
    raise ValueError(
        f"{location}: cancel-in-progress={cancel!r} can cancel non-PR (main/tag/nightly) runs"
    )


def validate_no_main_cancellation(doc, workflow_name):
    """Fail if a workflow or any job in it can cancel an in-progress run on main or non-PR triggers."""
    if not isinstance(doc, dict):
        raise ValueError(f"{workflow_name}: workflow document must be a mapping")
    _validate_concurrency_block(doc.get("concurrency"), workflow_name)
    jobs = doc.get("jobs") or {}
    if isinstance(jobs, dict):
        for job_name, job in jobs.items():
            if isinstance(job, dict) and "concurrency" in job:
                _validate_concurrency_block(job["concurrency"], f"{workflow_name}:jobs.{job_name}")


REQUIRED_JOBS = {
    "unit",
    "go",
    "tla (memmode)",
    "tla (blocks)",
    "tla (grouped)",
    "tla (traces)",
    "m5-gpu",
    "m5-serve",
}


def gh_json(*args):
    return json.loads(subprocess.check_output(["gh", *args], text=True))


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


if __name__ == "__main__":
    repo, sha = sys.argv[1:]
    print(f"Waiting for {', '.join(sorted(REQUIRED_JOBS))} on the main push of {sha}", flush=True)
    while True:
        try:
            url = completed_ci(repo, sha)
        except (ValueError, subprocess.CalledProcessError) as err:
            sys.exit(str(err))
        if url:
            print(f"Release CI passed: {url}")
            break
        time.sleep(30)  # GitHub bounds the wait with the gate job's timeout.
