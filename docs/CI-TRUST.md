# Inky CI trust and main protection

Hosted Inky CI runs on pull requests and main pushes. To qualify a ready,
up-to-date PR for M5, an administrator reviews the revision and comments
`/test-m5 <full PR head SHA>`. This approval works when the PR uses the same
GitHub identity. The main workflow checks administrator access, the exact
PR head, and the current test merge. It requires the reviewed head and test
merge to have identical Git trees. Update and review a new head after an
out-of-date or changed PR. Post a fresh comment for a retry.

M5 workers register only with a private control repository. They never
join public Inky's runner inventory. Public main CI dispatches M5 tests to
that repository and waits for authenticated results independently for GPU
and serving. The PR run publishes a single `m5-reviewed` receipt on the
approved head. The public main nightly dispatches the full release-profile
suite. Release verification retains the two original native main CI job
names and checks that both authenticated gates ran.

## Administrator cutover

1. Approve a private repository in `robotgamedevelopers` dedicated to Inky
   CI, or nominate an existing equivalent with access restricted to
   trusted CI administrators. Restrict changes and branches in that
   repository to those administrators and protect its default branch
   `main`. Install [the workflow template](../ci/inky-m5-control.yml)
   at `.github/workflows/inky-m5.yml`. The private repository is the
   runner access boundary. Review future changes to its workflow before
   promoting the updated template. [GitHub self-hosted runner access](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/manage-access)
2. Create two scoped Apps with webhooks disabled. Install the dispatch
   App only in the private control repository, with **Actions: write**.
   Install the result App only in public Inky, with **Checks: write**.
   Both inherit Metadata: read. Use separate private keys. In public
   Inky, create the `inky-m5-dispatch` environment with the single
   selected branch `main` and no tag access. Set environment variables
   `INKY_M5_DISPATCH_APP_ID` and `INKY_M5_CONTROL_REPOSITORY` (just the
   repository name). Store `INKY_M5_DISPATCH_APP_PRIVATE_KEY` as an
   environment secret. Set public Inky's repository variable
   `INKY_M5_REPORT_APP_ID` to the reporting App ID. [Dispatch permissions](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event)
3. In the private control repository, set the repository variable
   `INKY_M5_DISPATCH_ACTOR` to the dispatch App's bot login. Create the
   `inky-m5-report` environment with the single selected branch `main`
   and no tag access. Set environment variable `INKY_M5_REPORT_APP_ID`
   and environment secret `INKY_M5_REPORT_APP_PRIVATE_KEY`. The report
   job uses a GitHub-hosted runner. Neither private key is installed on
   Macs or passed into the test jobs. [Environment configuration](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)
4. Provision a standard, dedicated macOS service account on an approved
   M5 host with no passwordless sudo, access to the owner home, or
   existing project credentials. Register clean ephemeral ARM64 macOS
   workers to the **private control repository** with the `m5` label.
   Give each listener an independent workspace. Install Rust and the
   Metal runtime and provide read-only model and reference fixtures.
   The current test defaults are `~/inky-ci/ref` and
   `~/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf`.
   Verify both commands from under the dedicated account. Create a new
   clean listener registration after each ephemeral job.
   [Ephemeral runners](https://docs.github.com/en/actions/reference/runners/self-hosted-runners#ephemeral-runners-for-autoscaling)
5. After existing public M5 jobs drain, unregister all public Inky
   self-hosted runners, including offline registrations. Leave unrelated
   projects' listeners alone. Verify public Inky has no access to an
   organization M5 group. Use the temporary hosted checks in
   [Main ruleset](MAIN-RULESET.md) to merge the public workflow change.
   Confirm the two public main-push M5 jobs independently received the
   reporting App's gate results for the exact commit, run and attempt.
6. With a benign, up-to-date PR, verify an ordinary PR cannot enqueue
   private M5. Verify an administrator review command with a stale SHA
   is rejected. Confirm the valid command runs both short tests and
   publishes the dedicated App's `m5-reviewed` on the reviewed head.
   Rehearse a nightly dispatch from main and confirm it waits for its
   own full-suite receipt. Enable the final ruleset context only after
   this passes.

A failed cutover leaves public runner registrations disabled. Restore CI
by repairing the private receiver or reporting configuration and rerunning
the verification. To retry an unsuccessful main-push dispatch, rerun the
whole public workflow so it sends a new request.
