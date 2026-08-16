#!/usr/bin/env bash
set -euo pipefail

repo=/home/b/brain-mri-data
units="${HOME}/.config/systemd/user"

if [[ "$(id -un)" != b || "$(pwd -P)" != "$repo" ]]; then
  echo "Run as b from $repo" >&2
  exit 2
fi

install -d -m 700 runs/language-inbox/{ready,processing,processed,quarantine,explanations}
install -d -m 700 "$units"
install -m 644 scripts/systemd/brain-mri-language.path "$units/brain-mri-language.path"
install -m 644 scripts/systemd/brain-mri-language-consume.service "$units/brain-mri-language-consume.service"
systemctl --user daemon-reload
systemctl --user enable --now brain-mri-language.path
systemctl --user --no-pager status brain-mri-language.path
