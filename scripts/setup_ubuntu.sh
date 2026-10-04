#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv tesseract-ocr fonts-dejavu-core libglib2.0-0

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "Environment ready. Start with: ./scripts/start_ubuntu.sh"
