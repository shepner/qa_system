#!/bin/bash
# Reprocess all incomplete or partially processed files using the QA System CLI
# Usage: ./reprocess_incomplete_files.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
cd "$ROOT_DIR"

INCOMPLETE_FILES=$(./run.sh --list incomplete)

if [ -z "$INCOMPLETE_FILES" ]; then
  echo "No incomplete files found."
  exit 0
fi

echo "Found incomplete files:"
echo "$INCOMPLETE_FILES"
echo

for f in $INCOMPLETE_FILES; do
  echo "Reprocessing: $f"
  ./run.sh --add "$f"
done

echo "Reprocessing complete." 