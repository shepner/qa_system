#!/bin/bash
# Run the ChromaDB vector database example with Gemini embeddings

set -euo pipefail

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Load .env file if it exists in the project root or secrets directory
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
elif [ -f "secrets/.env" ]; then
  set -a
  source secrets/.env
  set +a
fi

python reference_example/chroma_vectordb_example.py 