# Main ruleset

Pull requests and pushes to main run `.github/workflows/ci.yml`: unit tests and a smoke test, nothing else
(Glenn, 2026-11-25), on GitHub-hosted runners, in about a minute. Everything else (the Go tool suites, every
TLA+ model and trace check, the slower unit extras, the release-profile build, and every GPU and serve gate on
the M5 Macs) runs in `.github/workflows/nightly.yml` on the fleet's self-hosted runners, and the dependency
audit runs daily in `.github/workflows/dependency-audit.yml`. Neither of those reports on a pull request, so
neither is a required check.

Create an active branch ruleset in **Settings → Rules → Rulesets**. Target
only `refs/heads/main`. Leave the bypass list empty and restrict deletion
and force pushes. Require a pull request with zero required approving reviews
and require status checks with **Require branches to be up to date before
merging** enabled. [GitHub ruleset options](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)

Select these exact status checks and select **GitHub Actions** as each
check's expected source. They are every job of `ci.yml`, and the same set
`scripts/release_ci.py` (`REQUIRED_JOBS`) waits for on the main push before a release.

- `unit`
- `smoke`

Both run on GitHub-hosted runners, so a fork pull request runs them too; no
self-hosted runner ever runs pull-request code.

Confirm the active rules using `gh api repos/robotgamedevelopers/inky/rules/branches/main`.
Verify a current pull request can merge after both checks pass and an
out-of-date pull request needs a branch update. A change to the jobs in
`ci.yml` needs the same change to the required contexts here, in the live
ruleset, and in `REQUIRED_JOBS`.
