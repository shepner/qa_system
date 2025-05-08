#!/bin/bash
# Run the Gemini talk-to-documents-with-embeddings example

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

# Reminder: Set GEMINI_API_KEY in your environment or update the Python file to use os.getenv("GEMINI_API_KEY")
python reference_example/talk_to_documents_with_embeddings_example.py 