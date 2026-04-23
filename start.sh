#!/bin/bash
set -e

# If the vector database does not exist, build it from the committed files
if [ ! -d "chroma_store" ]; then
  echo "==> chroma_store not found. Running pipeline setup..."
  python pipeline/embed.py
  python pipeline/extract_rules.py
  echo "==> Pipeline setup complete."
fi

# Start the API server on the port Railway provides
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
