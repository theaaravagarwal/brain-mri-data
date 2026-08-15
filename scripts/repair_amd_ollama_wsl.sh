#!/usr/bin/env bash
# Repair Ollama GPU discovery on the project's AMD ROCm WSL2 worker.
set -euo pipefail

if [[ ! -e /dev/dxg ]]; then
  echo "WSL GPU interface /dev/dxg is unavailable; fix the Windows AMD WSL driver first." >&2
  exit 1
fi

if ! command -v ollama >/dev/null; then
  echo "Ollama is not installed. Install its Linux AMD ROCm package before running this helper." >&2
  exit 1
fi

if ! rocminfo 2>/dev/null | grep -q 'gfx1100'; then
  echo "ROCm does not expose the RX 7900 XTX/XT as gfx1100; do not start Ollama yet." >&2
  exit 1
fi

echo "This updates only Ollama's systemd drop-in and saves a timestamped backup."
sudo -v

dropin_dir=/etc/systemd/system/ollama.service.d
dropin_path="$dropin_dir/wsl-gpu.conf"
backup_path="$dropin_path.bak.$(date +%Y%m%d%H%M%S)"
tmp_path=$(mktemp)
trap 'rm -f "$tmp_path"' EXIT

cat >"$tmp_path" <<'EOF'
[Service]
# WSL2 GPU passthrough is provided by AMD's WSL ROCm runtime. Do not preload a
# versioned libhsa path: package upgrades can leave that path stale and force
# Ollama's GPU-discovery subprocess to crash.
Environment="HSA_ENABLE_DXG_DETECTION=1"
Environment="OLLAMA_LLM_LIBRARY=rocm_v7_2"
Environment="OLLAMA_KEEP_ALIVE=5m"
Environment="OLLAMA_CONTEXT_LENGTH=8192"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
EOF

sudo install -d -m 0755 "$dropin_dir"
if sudo test -f "$dropin_path"; then
  sudo cp -a "$dropin_path" "$backup_path"
  echo "Saved previous drop-in: $backup_path"
fi
sudo install -m 0644 "$tmp_path" "$dropin_path"
sudo systemctl daemon-reload
sudo systemctl restart ollama

echo "Waiting for Ollama to restart..."
for _ in {1..15}; do
  if ollama --version >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Ollama server started. Run this short GPU check (it keeps the model warm for five minutes):"
echo '  ollama run qwen3:14b "Reply with exactly: OK"'
echo '  ollama ps'
echo
echo "Success requires the PROCESSOR column to include GPU. If it says CPU, stop the model"
echo "with 'ollama stop qwen3:14b' and inspect: journalctl -u ollama -n 120 --no-pager"
