# Main ruleset

In repository **Settings → Rules → Rulesets**, create an active branch
ruleset targeting only `refs/heads/main`. Use an empty bypass list. Restrict
deletion and force pushes. Require a pull request before merging with zero
required approving reviews. Require status checks and enable **Require
branches to be up to date before merging**. Use the following contexts and
verify the selected source of each. [GitHub ruleset options](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)

| Context | Expected source | Gate |
| --- | --- | --- |
| `unit` | GitHub Actions | CPU tests on a hosted Mac |
| `go` | GitHub Actions | Go tests on hosted Linux |
| `tla (memmode)` | GitHub Actions | Memory-mode models |
| `tla (blocks)` | GitHub Actions | Block models |
| `tla (grouped)` | GitHub Actions | Grouped models |
| `tla (traces)` | GitHub Actions | Trace checks |
| `m5-reviewed` | Dedicated Inky M5 App | Both the GPU and serving gates on M5 |

During the [M5 migration](CI-TRUST.md), switch the required contexts to
the hosted jobs for the cutover. Add the last required context only after smoke-testing a genuine
passing `m5-reviewed` on a ready, current PR. GitHub treats skipped native
jobs as successful required checks. The individual main-push `m5-gpu`
and `m5-serve` jobs therefore remain release checks, and the App receipt
is the required M5 check for merging a PR.
[GitHub required check troubleshooting](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)

Confirm the active rules with `gh api repos/robotgamedevelopers/inky/rules/branches/main`.
On an up-to-date test PR, verify GitHub blocks merging until the genuine
checks have all passed for the current head. Verify a newer head or an
out-of-date branch still blocks merging.
