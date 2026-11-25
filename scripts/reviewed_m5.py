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


def api(path, *args):
    return json.loads(subprocess.check_output(
        ["gh", "api", f"repos/{REPOSITORY}/{path}", *args], text=True,
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
    if permission["permission"] not in {"admin", "maintain", "write"}:
        raise ValueError("the reviewer must have repository write access")

    pull_path = f"pulls/{number}"
    pull = request(pull_path)
    if (pull["state"], pull["draft"], pull["base"]["repo"]["full_name"], pull["base"]["ref"]) != (
        "open", False, REPOSITORY, "main",
    ):
        raise ValueError("the pull request must be open, ready for review and targeting Inky main")
    if pull["user"]["login"].casefold() == reviewer.casefold():
        raise ValueError("the pull request author cannot authorize their own run")
    if pull["head"]["sha"] != approved_head:
        raise ValueError("the approved SHA must match the current pull request head")

    latest_review = None
    for page in request(f"{pull_path}/reviews?per_page=100", "--paginate", "--slurp"):
        for review in page:
            if (review["user"]["login"].casefold() == reviewer.casefold() and
                review["state"] in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}):
                latest_review = review
    if (latest_review is None or latest_review["state"] != "APPROVED" or
        latest_review["commit_id"] != approved_head):
        raise ValueError("submit a fresh approving review on the specified head before testing")

    candidate = pull["merge_commit_sha"]
    if pull["mergeable"] is not True or not candidate or not re.fullmatch(SHA, candidate):
        raise ValueError("GitHub has no merge candidate yet, retry after it computes mergeability")
    base = request("git/ref/heads/main")["object"]["sha"]
    parents = [parent["sha"] for parent in request(f"git/commits/{candidate}")["parents"]]
    if parents != [base, approved_head]:
        raise ValueError("the merge candidate does not combine current main and the approved head")
    # Fail closed if a fresh read sees the PR change while the candidate was being resolved.
    current = request(pull_path)
    if (current["state"], current["draft"], current["base"]["ref"],
        current["head"]["sha"], current["merge_commit_sha"]) != (
        "open", False, "main", approved_head, candidate,
    ):
        raise ValueError("the pull request changed while resolving the merge candidate")
    return candidate


def main():
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
        candidate = resolve_candidate(json.load(event_file))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        print(f"candidate={candidate}", file=output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
