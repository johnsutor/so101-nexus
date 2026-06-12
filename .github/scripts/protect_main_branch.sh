#!/usr/bin/env bash
set -euo pipefail

api_version="${GITHUB_API_VERSION:-2026-03-10}"
token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
branch="${BRANCH:-main}"

remote_url="$(git config --get remote.origin.url)"
owner_repo="${remote_url#git@github.com:}"
owner_repo="${owner_repo#https://github.com/}"
owner_repo="${owner_repo%.git}"

owner="${GITHUB_OWNER:-${OWNER:-${owner_repo%%/*}}}"
repo="${GITHUB_REPO:-${REPO:-${owner_repo#*/}}}"

if [[ -z "$token" ]]; then
  cat >&2 <<'EOF'
Set GITHUB_TOKEN or GH_TOKEN to a token with Administration: write for this repository.
For a fine-grained PAT, grant Repository administration: read and write.
EOF
  exit 2
fi

api() {
  local method="$1"
  local path="$2"
  shift 2

  curl -fsSL \
    -X "$method" \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${token}" \
    -H "X-GitHub-Api-Version: ${api_version}" \
    "https://api.github.com${path}" \
    "$@"
}

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cat >"$tmpdir/actions-permissions.json" <<'JSON'
{
  "enabled": true,
  "allowed_actions": "selected",
  "sha_pinning_required": true
}
JSON

cat >"$tmpdir/selected-actions.json" <<'JSON'
{
  "github_owned_allowed": true,
  "verified_allowed": false,
  "patterns_allowed": [
    "astral-sh/setup-uv@*",
    "pnpm/action-setup@*",
    "pypa/gh-action-pypi-publish@*"
  ]
}
JSON

cat >"$tmpdir/workflow-permissions.json" <<'JSON'
{
  "default_workflow_permissions": "read",
  "can_approve_pull_request_reviews": false
}
JSON

cat >"$tmpdir/fork-pr-approval.json" <<'JSON'
{
  "approval_policy": "all_external_contributors"
}
JSON

cat >"$tmpdir/branch-protection.json" <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "lint",
      "test (core)",
      "test (mujoco)",
      "test (maniskill)",
      "coverage",
      "docs",
      "dependency-review"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1,
    "require_last_push_approval": true
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}
JSON

echo "Applying repository Actions policy for ${owner}/${repo}"
api PUT "/repos/${owner}/${repo}/actions/permissions" \
  -d @"$tmpdir/actions-permissions.json" >/dev/null

echo "Allowing only GitHub-owned actions and explicit third-party action repositories"
api PUT "/repos/${owner}/${repo}/actions/permissions/selected-actions" \
  -d @"$tmpdir/selected-actions.json" >/dev/null

echo "Setting the default GITHUB_TOKEN permission to read-only"
api PUT "/repos/${owner}/${repo}/actions/permissions/workflow" \
  -d @"$tmpdir/workflow-permissions.json" >/dev/null

echo "Requiring approval for all external fork pull request workflows"
api PUT "/repos/${owner}/${repo}/actions/permissions/fork-pr-contributor-approval" \
  -d @"$tmpdir/fork-pr-approval.json" >/dev/null

echo "Applying branch protection to ${branch}"
api PUT "/repos/${owner}/${repo}/branches/${branch}/protection" \
  -d @"$tmpdir/branch-protection.json" >/dev/null

echo "Requiring signed commits on ${branch}"
api POST "/repos/${owner}/${repo}/branches/${branch}/protection/required_signatures" >/dev/null

echo "Repository Actions policy and ${branch} branch protection are configured."
