"""Wait for the tagged commit's main-push CI before publishing any release assets."""
import json
import subprocess
import sys
import time


REQUIRED_JOBS = {"cpu", "go", "m5"}


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
    print(f"Waiting for cpu, go and m5 on the main push of {sha}", flush=True)
    while True:
        try:
            url = completed_ci(repo, sha)
        except (ValueError, subprocess.CalledProcessError) as err:
            sys.exit(str(err))
        if url:
            print(f"Release CI passed: {url}")
            break
        time.sleep(30)  # GitHub bounds the wait with the gate job's timeout.
