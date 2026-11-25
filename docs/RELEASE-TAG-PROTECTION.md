# Release Tag Protection & `CRATES_IO_TOKEN` Scoping

This specification pairs repository-side GitHub tag rulesets and Actions environment protection with the workflow guards enforced in `.github/workflows/release.yml` and `scripts/release_ci.py` (`U43`, `U132`, `U133`).

## 1. Workflow-Enforced Invariants (`.github/workflows/release.yml`)

Every release workflow run enforces all four gates in `scripts/release_ci.py`:

1. **Tag Ref & Ancestry (`U43`)**: `GITHUB_REF` must match `refs/tags/vX.Y.Z*` (`GITHUB_REF == "refs/tags/$GITHUB_REF_NAME"`), its dereferenced commit `TAG_SHA` must match `origin`'s `refs/tags/$GITHUB_REF_NAME`, and `git merge-base --is-ancestor "$TAG_SHA" origin/main` must hold before waiting for CI or building assets.
2. **Manifest & Notes Match**: `[package] version` in `Cargo.toml` must equal `${GITHUB_REF_NAME#v}`, and `docs/releases/$GITHUB_REF_NAME.md` must exist and be non-empty.
3. **Fail-Loud `CRATES_IO_TOKEN` (`U132`)**: `release` runs inside `environment: crates-io-release` and aborts up front before running tests or builds if `CRATES_IO_TOKEN` is unset or empty.
4. **Remote Tag Immutability Before Publish & Release (`U43`)**: Immediately before `gh release create` **and** immediately before `cargo publish --locked`, `scripts/release_ci.py --verify-remote-tag origin "$TAG" "$TAG_SHA"` re-queries `git ls-remote origin "refs/tags/$TAG" "refs/tags/$TAG^{}"` and aborts if the remote tag disappeared or was force-moved during the build.

## 2. GitHub Actions Environment (`crates-io-release`, `U133`)

Store `CRATES_IO_TOKEN` as an **Environment secret** inside `crates-io-release` (never as an unscoped repository-wide secret so non-release workflows cannot read it):

- **Environment name**: `crates-io-release`
- **Deployment branches and tags**: *Selected branches and tags*
  - Add **Tag pattern**: `v[0-9]*` (no branch patterns allowed)
- **Environment secrets**: `CRATES_IO_TOKEN`

## 3. Repository Tag Protection Ruleset (`refs/tags/v*`, `U43`)

Apply the following repository ruleset on `robotgamedevelopers/inky` so release tags `refs/tags/v*` can only be created by repository admins and can never be force-pushed (`non_fast_forward`) or deleted (`deletion`):

```json
{
  "name": "protect-release-tags",
  "target": "tag",
  "enforcement": "active",
  "bypass_actors": [
    {
      "actor_id": 5,
      "actor_type": "RepositoryRole",
      "bypass_mode": "always"
    }
  ],
  "conditions": {
    "ref_name": {
      "include": ["refs/tags/v*"],
      "exclude": []
    }
  },
  "rules": [
    { "type": "creation" },
    { "type": "update" },
    { "type": "deletion" },
    { "type": "non_fast_forward" }
  ]
}
```

Apply or inspect with `gh`:

```sh
gh api repos/robotgamedevelopers/inky/rulesets --method POST --input ruleset.json
gh api repos/robotgamedevelopers/inky/rulesets
```
