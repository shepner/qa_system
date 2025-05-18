#!/bin/bash
# Run Gemini image understanding reference example
# Usage: bash run_gemini_image_understanding_example.sh [image_path]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Load .env file if it exists in the secrets directory (consistent with other run scripts)
if [ -f "secrets/.env" ]; then
  set -a
  source secrets/.env
  set +a
fi

# Accept image path as argument, default to reference_example/sample.jpg
if [ $# -ge 1 ]; then
  IMAGE_PATH="$1"
else
  IMAGE_PATH="reference_example/sample.jpg"
fi

echo "Using image: $IMAGE_PATH"
if [ ! -f "$IMAGE_PATH" ]; then
  echo "Error: $IMAGE_PATH not found. Please provide a valid image file." >&2
  exit 1
fi

# Run the example
python reference_example/gemini_image_understanding_example.py "$IMAGE_PATH" 