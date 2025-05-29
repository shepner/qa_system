#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
cd "$ROOT_DIR"

FILES=$(./run.sh --list "*.pdf")

if [ -z "$FILES" ]; then
  echo "No pdf files found."
  exit 0
fi

echo "Found pdf files:"
printf '%s\n' "$FILES"
echo

# Read each file line by line, preserving spaces
while IFS= read -r f; do
  if [ -n "$f" ]; then
    echo "Removing: $f"
    ./run.sh --remove "$f"
  fi
done <<< "$FILES"

echo "complete." 