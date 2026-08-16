#!/usr/bin/env bash
# Configure Ollama to use the project's AMD ROCm WSL2 worker safely.
set -euo pipefail

if [[ ! -e /dev/dxg ]]; then
  echo "WSL GPU interface /dev/dxg is unavailable; fix the Windows AMD WSL driver first." >&2
  exit 1
fi

if ! command -v ollama >/dev/null; then
  echo "Ollama is not installed. Install its Linux AMD ROCm package before running this helper." >&2
  exit 1
fi

gpu_ready=false
set +e
rocminfo_output="$(env -u HSA_ENABLE_DXG_DETECTION rocminfo 2>&1)"
rocminfo_status=$?
set -e
if grep -q 'gfx1100' <<<"$rocminfo_output"; then
  gpu_ready=true
elif [[ -x .venv/bin/python ]] && env -u HSA_ENABLE_DXG_DETECTION .venv/bin/python - <<'PY'
import torch
assert torch.version.hip is not None
assert torch.cuda.is_available()
values = torch.ones((32, 32), device="cuda")
assert values.sum().item() == 1024
PY
then
  echo "rocminfo was unreliable, but a real ROCm PyTorch tensor check passed."
  gpu_ready=true
fi
if [[ "$gpu_ready" != true ]]; then
  if [[ $rocminfo_status -ne 0 ]]; then
    echo "rocminfo failed while checking the RX 7900 XT/XTX." >&2
  fi
  echo "Neither rocminfo nor a real ROCm compute check can see gfx1100; do not start Ollama yet." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
unit_path="$unit_dir/ollama.service"
backup_path="$unit_path.bak.$(date +%Y%m%d%H%M%S)"
log_dir="$repo_root/runs/logs"
log_path="$log_dir/ollama-amd.log"
tmp_path=$(mktemp)
trap 'rm -f "$tmp_path"' EXIT

mkdir -p "$unit_dir" "$log_dir"
touch "$log_path"
previous_log_lines=$(wc -l <"$log_path")

cat >"$tmp_path" <<EOF
[Unit]
Description=Ollama AMD ROCm worker for brain-mri-data
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$repo_root
# Ollama's bundled native-Linux HSA runtime does not enumerate WSL /dev/dxg.
# Preload the generic symlink to the installed WSL runtime that already passes
# the project's PyTorch compute check. Keep DXG detection overrides unset.
ExecStart=/usr/bin/env -u HSA_ENABLE_DXG_DETECTION LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/usr/local/lib/ollama/rocm_v7_2 OLLAMA_LLM_LIBRARY=rocm_v7_2 OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=5m OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 /usr/local/bin/ollama serve
Restart=on-failure
RestartSec=3
StandardOutput=append:$log_path
StandardError=append:$log_path

[Install]
WantedBy=default.target
EOF

if [[ -f "$unit_path" ]]; then
  cp -a "$unit_path" "$backup_path"
  echo "Saved previous user unit: $backup_path"
fi
install -m 0644 "$tmp_path" "$unit_path"
systemctl --user daemon-reload
systemctl --user enable ollama.service
systemctl --user restart ollama.service

if ! loginctl enable-linger "$(id -un)"; then
  echo "Warning: could not enable user lingering; Ollama may stop after logout." >&2
fi

echo "Waiting for Ollama to enumerate the GPU..."
for _ in {1..15}; do
  if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

new_log="$(tail -n "+$((previous_log_lines + 1))" "$log_path")"
if ! grep -Eq 'library=ROCm.*compute=gfx1100' <<<"$new_log"; then
  systemctl --user stop ollama.service
  echo "Ollama did not enumerate ROCm gfx1100; service stopped to prevent CPU fallback." >&2
  echo "$new_log" >&2
  exit 1
fi

echo "Ollama is running through ROCm gfx1100. Verify every loaded model with:"
echo '  ollama ps'
echo "The PROCESSOR column must include GPU. Stop any CPU-backed model immediately."
