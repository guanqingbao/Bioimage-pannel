#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python_bin="$project_dir/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "Run ./scripts/setup_ubuntu.sh first." >&2
  exit 1
fi

export IMAGE_BATCH_DATA_DIR="${IMAGE_BATCH_DATA_DIR:-$project_dir/data}"
host="${IMAGE_BATCH_HOST:-127.0.0.1}"
port="${IMAGE_BATCH_PORT:-8010}"

mkdir -p "$IMAGE_BATCH_DATA_DIR"
exec "$python_bin" -m uvicorn app:app --host "$host" --port "$port"
