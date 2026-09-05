#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-deploy}
host=software@100.64.0.7
base=/home/software/Documents/.aa/brain
if [[ "$mode" == rollback ]]; then
  ssh "$host" bash -s -- rollback < scripts/activate_prototype_release.sh
  exit
fi
[[ "$mode" == deploy ]] || { echo 'Usage: bash scripts/deploy_prototype.sh [deploy|rollback] [commit]'; exit 2; }
revision=$(git rev-parse "${2:-HEAD}^{commit}")
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || exit 2
bundle=$(mktemp /tmp/brain-release.XXXXXX)
trap 'unlink "$bundle"' EXIT
git archive --format=tar -o "$bundle" "$revision"
scp "$bundle" "$host:/tmp/brain-release-$revision.tar"
ssh "$host" bash -s -- "$revision" < scripts/activate_prototype_release.sh
if ! node scripts/check_prototype.mjs http://100.64.0.1:4173 --require-llm || ! node scripts/check_prototype.mjs http://100.64.0.1:4173 --evaluation --require-llm; then
  echo 'Acceptance failed; restoring previous release.' >&2
  ssh "$host" bash -s -- rollback < scripts/activate_prototype_release.sh
  exit 1
fi
echo "Deployed and verified $revision"
