#!/usr/bin/env bash
# Invoked over SSH by deploy_prototype.sh on .7 only.
set -euo pipefail
base=/home/software/Documents/.aa/brain
mode=$1
if pgrep -f '[r]un_4060_research_inference.py' >/dev/null; then
  echo 'Inference is active; retry deployment after it finishes.' >&2
  exit 1
fi
mkdir -p "$base/deploy/releases"
if [[ "$mode" == rollback ]]; then
  target=$(readlink -f "$base/deploy/previous")
  [[ "$target" == "$base" || "$target" == "$base"/deploy/releases/* ]] || exit 2
  ln -s "$target" "$base/deploy/current.next"
  mv -Tf "$base/deploy/current.next" "$base/deploy/current"
else
  [[ "$mode" =~ ^[0-9a-f]{40}$ ]] || exit 2
  release="$base/deploy/releases/$mode"
  [[ ! -e "$release" ]] || { echo 'Release directory already exists; use a new commit.' >&2; exit 1; }
  mkdir "$release"
  tar -xf "/tmp/brain-release-$mode.tar" -C "$release"
  unlink "/tmp/brain-release-$mode.tar"
  for directory in .venv .tools runs artifacts; do
    ln -s "$base/$directory" "$release/$directory"
  done
  ln -s "$base/monitor/.runtime" "$release/monitor/.runtime"
  export PATH="$base/.tools/node/bin:$PATH"
  npm --prefix "$release/monitor" ci
  npm --prefix "$release/monitor" run build
  if pgrep -f '[r]un_4060_research_inference.py' >/dev/null; then echo 'Inference started during build; activation cancelled.' >&2; exit 1; fi
  previous=$(readlink -f "$base/deploy/current" || true)
  [[ -n "$previous" && -d "$previous" ]] || previous="$base"
  ln -sfn "$previous" "$base/deploy/previous"
  ln -s "$release" "$base/deploy/current.next"
  mv -Tf "$base/deploy/current.next" "$base/deploy/current"
  mkdir -p /home/software/.config/systemd/user/brain-mri-prototype.service.d
  install -m 644 "$release/scripts/systemd/prototype-release.conf" /home/software/.config/systemd/user/brain-mri-prototype.service.d/release.conf
fi
systemctl --user daemon-reload
systemctl --user restart brain-mri-prototype
for attempt in {1..30}; do
  if curl -fsS http://100.64.0.7:4173/api/capabilities | /home/software/Documents/.aa/brain/.tools/node/bin/node -e 'let s="";process.stdin.on("data",c=>s+=c);process.stdin.on("end",()=>{try{process.exit(JSON.parse(s).inference.status==="ready"?0:1)}catch{process.exit(1)}})'; then exit 0; fi
  sleep 2
done
echo 'Application did not become ready.' >&2
target=$(readlink -f "$base/deploy/previous")
ln -s "$target" "$base/deploy/current.next"
mv -Tf "$base/deploy/current.next" "$base/deploy/current"
systemctl --user restart brain-mri-prototype
exit 1
