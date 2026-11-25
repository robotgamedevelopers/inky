# Main ruleset with native CI

After the `m5-pr-result` job has passed on a same-repository pull request,
create an active branch ruleset in **Settings → Rules → Rulesets**. Target
only `refs/heads/main`. Leave the bypass list empty and restrict deletion
and force pushes. Require a pull request with zero required approving reviews
and require status checks with **Require branches to be up to date before
merging** enabled. [GitHub ruleset options](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)

Select these exact status checks and select **GitHub Actions** as each
check's expected source.

- `unit`
- `go`
- `tla (memmode)`
- `tla (blocks)`
- `tla (grouped)`
- `tla (traces)`
- `m5-gpu`
- `m5-serve`
- `m5-pr-result`

The hosted `m5-pr-result` job checks that both native M5 jobs actually
succeeded. GitHub's required status rule treats a skipped job as successful,
so the two native M5 jobs alone would not enforce this. A fork PR needs a
maintainer to stage its reviewed change on an up-to-date branch inside
Inky and open a same-repository PR.
[GitHub required check troubleshooting](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)

Confirm the active rules using `gh api repos/robotgamedevelopers/inky/rules/branches/main`.
Verify a current same-repository PR can merge after all checks pass, an
out-of-date PR needs a branch update, and a fork PR with skipped M5 checks is
blocked. This ruleset covers the native CI topology. Changes to self-hosted
runner access or M5 triggers require a corresponding change to the required
M5 contexts.
