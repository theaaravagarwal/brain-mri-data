#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
node_version="24.20.0"
archive="node-v${node_version}-linux-x64.tar.xz"
expected_sha256="2f2c0da162318f0de47665410c7c8c2ed3d36c8f3105de4bbc61176c70a7cbf2"
tools_dir="$repo_root/.tools"
install_dir="$tools_dir/node-v${node_version}-linux-x64"
active_link="$tools_dir/node"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This pinned runtime installer supports Linux x86_64 only." >&2
  exit 64
fi
if [[ -x "$install_dir/bin/node" ]]; then
  ln -sfn "$(basename "$install_dir")" "$active_link"
  export PATH="$active_link/bin:$PATH"
  "$active_link/bin/node" --version
  "$active_link/bin/npm" --version
  exit 0
fi

temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT
curl --proto '=https' --tlsv1.2 -fsSLo "$temporary_dir/$archive" "https://nodejs.org/dist/v${node_version}/$archive"
printf '%s  %s\n' "$expected_sha256" "$temporary_dir/$archive" | sha256sum --check --status
mkdir -p "$tools_dir"
tar -xJf "$temporary_dir/$archive" -C "$tools_dir"
ln -sfn "$(basename "$install_dir")" "$active_link"
export PATH="$active_link/bin:$PATH"
"$active_link/bin/node" --version
"$active_link/bin/npm" --version
