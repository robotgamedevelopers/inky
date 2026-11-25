"""Fail fast with an explicit Actions error when no M5 self-hosted runner is online."""
import argparse
import json
import subprocess
import sys
import time


REQUIRED_LABELS = {"self-hosted", "macos", "arm64", "m5"}
KNOWN_M5_JOBS = {"m5-gpu", "m5-serve", "m5-full"}


class GhHttpError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def gh_api_get(endpoint):
    proc = subprocess.run(
        ["gh", "api", "-i", endpoint],
        capture_output=True,
        text=True,
        check=False,
    )
    raw = proc.stdout or ""
    status = None
    body = raw
    if "\r\n\r\n" in raw:
        head, body = raw.split("\r\n\r\n", 1)
    elif "\n\n" in raw:
        head, body = raw.split("\n\n", 1)
    else:
        head = ""
    if head:
        first = head.splitlines()[0].split()
        if len(first) >= 2 and first[1].isdigit():
            status = int(first[1])
    if proc.returncode != 0:
        if status is None:
            m = proc.stderr or body
            for code in (401, 403, 404):
                if f"HTTP {code}" in m or f"status {code}" in m:
                    status = code
                    break
        raise GhHttpError(status or 500, (proc.stderr or body).strip())
    return json.loads(body) if body.strip() else {}


def gh_api_post(endpoint):
    subprocess.run(
        ["gh", "api", "-X", "POST", endpoint],
        capture_output=True,
        text=True,
        check=False,
    )


def online_m5_runners(runners_payload):
    online = []
    for r in runners_payload.get("runners", []):
        if r.get("status") != "online":
            continue
        labels = {lbl.get("name", "").lower() for lbl in r.get("labels", []) if isinstance(lbl, dict)}
        if REQUIRED_LABELS.issubset(labels):
            online.append(r.get("name", "m5-runner"))
    return online


def is_m5_job(job, target_names):
    name = job.get("name", "")
    if name in target_names or name in KNOWN_M5_JOBS:
        return True
    labels = {str(lbl).lower() for lbl in job.get("labels", [])}
    return "m5" in labels


def check_m5_availability(
    repo,
    run_id,
    target_jobs,
    max_queue_seconds=90.0,
    poll_seconds=5.0,
    cancel_on_offline=False,
    api_get=gh_api_get,
    api_post=gh_api_post,
    clock=time.monotonic,
    sleep_fn=time.sleep,
    out=sys.stdout,
):
    target_set = {j.strip() for j in target_jobs if j.strip()}
    try:
        payload = api_get(f"repos/{repo}/actions/runners?per_page=100")
    except GhHttpError as err:
        if err.status not in (401, 403, 404):
            raise
    else:
        online = online_m5_runners(payload)
        if online:
            print(f"M5 self-hosted runner(s) online: {', '.join(online)}", file=out, flush=True)
            return 0
        print(
            "::error::No online M5 self-hosted runner registered for [self-hosted, macOS, ARM64, m5] (0 online)",
            file=out,
            flush=True,
        )
        if cancel_on_offline and run_id:
            api_post(f"repos/{repo}/actions/runs/{run_id}/cancel")
        return 1

    # Standard GITHUB_TOKEN lacks Administration:read for /actions/runners (HTTP 403),
    # while actions:read always works. Watch our run's M5 jobs alongside repository-wide
    # in_progress runs: if an M5 job stays queued past max_queue_seconds while 0 M5 jobs
    # are active across the repository, zero M5 runners are online.
    deadline = clock() + float(max_queue_seconds)
    run_id_int = int(run_id)
    while True:
        own = api_get(f"repos/{repo}/actions/runs/{run_id_int}/jobs?per_page=100")
        own_m5 = [j for j in own.get("jobs", []) if j.get("name") in target_set]
        if own_m5:
            if any(j.get("status") == "in_progress" for j in own_m5):
                picked = [j["name"] for j in own_m5 if j.get("status") == "in_progress"]
                print(f"M5 runner picked up {', '.join(picked)}", file=out, flush=True)
                return 0
            if all(j.get("status") == "completed" for j in own_m5):
                print("Target M5 job(s) already completed.", file=out, flush=True)
                return 0

        in_prog = api_get(f"repos/{repo}/actions/runs?status=in_progress&per_page=50")
        active_m5 = 0
        for run in in_prog.get("workflow_runs", []):
            other_id = run.get("id")
            if not other_id or int(other_id) == run_id_int:
                continue
            other_jobs = api_get(f"repos/{repo}/actions/runs/{other_id}/jobs?per_page=100")
            for j in other_jobs.get("jobs", []):
                if j.get("status") == "in_progress" and is_m5_job(j, target_set):
                    active_m5 += 1

        if active_m5 > 0:
            print(
                f"M5 runner is online and busy ({active_m5} active M5 job(s)); queued jobs will wait normally.",
                file=out,
                flush=True,
            )
            return 0

        now = clock()
        if now >= deadline:
            queued_names = ", ".join(j.get("name", "m5") for j in own_m5) or ", ".join(sorted(target_set))
            print(
                f"::error::No online M5 self-hosted runner picked up the job (0 active M5 runs; queued: {queued_names})",
                file=out,
                flush=True,
            )
            if cancel_on_offline:
                api_post(f"repos/{repo}/actions/runs/{run_id_int}/cancel")
            return 1

        sleep_fn(min(float(poll_seconds), max(0.0, deadline - now)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--run-id", required=True, help="current workflow run database ID")
    parser.add_argument("--jobs", required=True, help="comma-separated M5 job names in this workflow")
    parser.add_argument("--max-queue-seconds", type=float, default=90.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--cancel-on-offline", action="store_true")
    args = parser.parse_args(argv)
    return check_m5_availability(
        repo=args.repo,
        run_id=args.run_id,
        target_jobs=args.jobs.split(","),
        max_queue_seconds=args.max_queue_seconds,
        poll_seconds=args.poll_seconds,
        cancel_on_offline=args.cancel_on_offline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
