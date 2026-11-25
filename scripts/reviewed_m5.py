"""Resolve the exact approved merge candidate for a trusted /test-m5 comment."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import quote


REPOSITORY = "robotgamedevelopers/inky"
SHA = r"[0-9a-f]{40}"


def api(path):
    return json.loads(subprocess.check_output(
        ["gh", "api", f"repos/{REPOSITORY}/{path}"], text=True,
    ))


def resolve_candidate(event, request=api):
    match = re.fullmatch(rf"/test-m5 ({SHA})", event["comment"]["body"].strip())
    if match is None:
        raise ValueError("use /test-m5 followed by the full approved PR head SHA")
    approved_head = match[1]
    reviewer = event["comment"]["user"]["login"]
    if event["comment"]["user"]["type"] != "User":
        raise ValueError("the command must come from a human reviewer")
    number = event["issue"]["number"]
    if type(number) is not int or number < 1 or not event["issue"].get("pull_request"):
        raise ValueError("the command must be on a pull request")

    permission = request(f"collaborators/{quote(reviewer, safe='')}/permission")
    if permission["permission"] != "admin":
        raise ValueError("the approval command requires repository administrator access")

    pull_path = f"pulls/{number}"
    pull = request(pull_path)
    if (pull["state"], pull["draft"], pull["base"]["repo"]["full_name"], pull["base"]["ref"]) != (
        "open", False, REPOSITORY, "main",
    ):
        raise ValueError("the pull request must be open, ready for review and targeting Inky main")
    if pull["head"]["sha"] != approved_head:
        raise ValueError("the approved SHA must match the current pull request head")

    candidate = pull["merge_commit_sha"]
    if pull["mergeable"] is not True or not candidate or not re.fullmatch(SHA, candidate):
        raise ValueError("GitHub has no merge candidate yet, retry after it computes mergeability")
    base = request("git/ref/heads/main")["object"]["sha"]
    relation = request(f"compare/{base}...{approved_head}?per_page=1")
    if relation["behind_by"] != 0 or relation["merge_base_commit"]["sha"] != base:
        raise ValueError("update the pull request head to include current main before reviewing")
    merge = request(f"git/commits/{candidate}")
    parents = [parent["sha"] for parent in merge["parents"]]
    if parents != [base, approved_head]:
        raise ValueError("the merge candidate does not combine current main and the approved head")
    head = request(f"git/commits/{approved_head}")
    if merge["tree"]["sha"] != head["tree"]["sha"]:
        raise ValueError("the reviewed head and merge candidate have different trees")
    # Fail closed if a fresh read sees the PR change while the candidate was being resolved.
    current = request(pull_path)
    if (current["state"], current["draft"], current["base"]["ref"],
        current["head"]["sha"], current["merge_commit_sha"]) != (
        "open", False, "main", approved_head, candidate,
    ):
        raise ValueError("the pull request changed while resolving the merge candidate")
    return approved_head, candidate


def main():
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
        head, candidate = resolve_candidate(json.load(event_file))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        print(f"head={head}\ncandidate={candidate}", file=output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
