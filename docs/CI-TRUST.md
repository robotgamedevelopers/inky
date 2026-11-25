# Inky CI trust and main protection

GitHub-hosted `unit`, `go`, and the four `tla` checks run on ordinary pull
requests and main pushes. A repository administrator reviews the exact
revision of an up-to-date PR to main and comments `/test-m5 <full PR head SHA>`.
This comment is the approval and works when the PR was opened under the
administrator's own GitHub identity. Post it from the administrator account
after reviewing and use a fresh comment for a retry.

The main workflow checks the administrator's current permission and confirms
the requested SHA is still the PR head. It builds GitHub's immutable merge
candidate only if it includes current main and has the same Git tree as the
approved head. The hosted check runs and the M5 receipt are both attached to
the PR head. If advancing main renders the branch out of date, update the
branch and review its new head before posting another approval.

The two M5 jobs also run on main pushes. Scheduled or manually dispatched
nightlies use the M5 runner group from main only. Manual dispatch on main is
`gh workflow run nightly.yml --ref main`. Release publication waits for the
actual main-push CI and its individual jobs.

## Administrator cutover

1. Create an organization runner group named `inky-reviewed-m5`. Allow access
   from public repositories and select only `robotgamedevelopers/inky`.
   Enable selected-workflow access and allow exactly these qualified refs.
   [Runner group configuration](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/manage-access)

   - `robotgamedevelopers/inky/.github/workflows/ci.yml@refs/heads/main`
   - `robotgamedevelopers/inky/.github/workflows/nightly.yml@refs/heads/main`

2. Create a private organization-owned GitHub App with repository Checks:
   read and write, and the implicit Metadata: read permission. Install it
   only on Inky. Create the `inky-m5-attestation` Actions environment with
   a selected **branch** rule for `main` and no tag rules. Put its App ID in
   the environment variable `INKY_M5_APP_ID` and its private key in the
   environment secret `INKY_M5_APP_PRIVATE_KEY`. The dedicated App identity
   lets the rule distinguish the M5 receipt from checks other Actions jobs
   can publish. [Environment configuration](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)

3. Provision a dedicated standard macOS service account on each M5 host and
   register fresh organization runners in the selected group with the `m5`
   label. Give each listener its own runner and work directories under the
   dedicated account. Install the Rust and Metal runtime prerequisites and
   provide fixtures through read-only paths. Verify the listener UID is
   outside the admin group, has no passwordless sudo, and runs with its own
   home and keychain. Restrict access to the owner home, credentials, other
   runner workspaces, and writable shared directories. Test the GPU gates
   from that account before routing production CI to it.

4. Coordinate the cutover after any outstanding old-workflow jobs drain.
   Disable and unregister all Inky repository-scoped runners, including
   offline registrations. Leave unrelated projects' listeners alone.
   Confirm other organization groups available to Inky contain no
   general-purpose self-hosted runners. Protect main with the temporary
   contexts described in [Main ruleset](MAIN-RULESET.md), then merge this
   workflow through that rule. Confirm the resulting main-push `m5-gpu` and
   `m5-serve` jobs use the new group and complete successfully.

5. From a benign, up-to-date PR, run hosted CI and have a repository
   administrator review and comment its exact head SHA. Verify both M5 jobs checked out GitHub's
   same test merge commit, passed on the dedicated account, and that
   `m5-reviewed` on the PR head was published by the dedicated App. Confirm
   ordinary PR events and non-main dispatches do not get M5 runners, and
   approval for an earlier head produces no candidate. Then add
   `m5-reviewed` to the required checks and verify merging is blocked
   before the authentic check and allowed after it. GitHub requires a
   recent successful check before treating it as an eligible required
   context. [Required check troubleshooting](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)

A failed cutover leaves the unsafe registrations disabled. Restore CI by
repairing the restricted group or attestations and rerunning the same
verification before enabling the final check.

See [Main ruleset](MAIN-RULESET.md) for the required checks and verification.
