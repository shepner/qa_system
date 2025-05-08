#!/bin/bash
# Script to run embedding_example.py using the current Python environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Install requirements before running the script
pip install -r requirements.txt

# Load .env file if it exists in the secrets directory
if [ -f "secrets/.env" ]; then
  set -a
  source secrets/.env
  set +a
fi

python reference_example/embedding_example.py 