"""Dispatch trusted M5 builds and verify the public App receipts."""
import json
import os
import re
import subprocess
import sys
import time


def api(path, *options):
    result = subprocess.check_output(["gh", "api", path, *options], text=True)
    return json.loads(result) if result.strip() else None


def dispatch(mode, head, candidate):
    control = os.environ["M5_CONTROL_REPOSITORY"]
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", control) or control.lower() == "inky":
        raise ValueError("configure a separate private Inky M5 control repository")
    request = f'{os.environ["GITHUB_RUN_ID"]}.{os.environ["GITHUB_RUN_ATTEMPT"]}'
    api(f"repos/robotgamedevelopers/{control}/actions/workflows/inky-m5.yml/dispatches",
        "-f", "ref=main", "-f", f"inputs[mode]={mode}", "-f", f"inputs[head]={head}",
        "-f", f"inputs[candidate]={candidate}", "-f", f"inputs[request]={request}")
    print(f"request={request}", file=sys.stdout)


def receipt(checks, app, head, name, request):
    matching = [
        check for page in checks for check in page["check_runs"]
        if check["name"] == name and check["head_sha"] == head and check["external_id"] == request
        and check["app"] and check["app"]["id"] == app and check["status"] == "completed"
    ]
    if matching:
        return max(matching, key=lambda check: check["id"])


def wait(name, head, request):
    app = int(os.environ["M5_REPORT_APP_ID"])
    while True:
        result = receipt(api(
            f"repos/robotgamedevelopers/inky/commits/{head}/check-runs"
            f"?check_name={name}&filter=all&per_page=100",
            "--paginate", "--slurp",
        ), app, head, name, request)
        if result is not None:
            if result["conclusion"] != "success":
                raise ValueError(f'{name} concluded {result["conclusion"]} in request {request}')
            return
        time.sleep(15)  # GitHub bounds the wait with the job timeout.


if __name__ == "__main__":
    try:
        command, *args = sys.argv[1:]
        {"dispatch": dispatch, "wait": wait}[command](*args)
    except (ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
