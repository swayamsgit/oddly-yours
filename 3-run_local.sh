#!/usr/bin/env bash
# One-command local start. Run from the package root (the folder with app/ data/ pipeline/).
set -e
cd "$(dirname "$0")/.."
[ -d venv ] || python3 -m venv venv
venv/bin/pip install -q -r app/requirements.txt
if [ -z "$ODDLY_LLM" ] && [ ! -f models/qwen.gguf ]; then
  echo "Optional local model not found. 'Like this, but' will use the rule-based planner."
  echo "To enable it: mkdir -p models && curl -L -o models/qwen.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
fi
export ODDLY_DATA="$PWD/data" ODDLY_LLM="${ODDLY_LLM:-$PWD/models/qwen.gguf}" PYTHONPATH="$PWD/app"
export GOOGLE_CLIENT_ID="${GOOGLE_CLIENT_ID:-559787477302-rukdtd9a1h4bgg3m2eufst4prajrbjcu.apps.googleusercontent.com}"
echo "Open http://localhost:8000 (Google sign-in only works on http://localhost:8000 or http://localhost)"
cd app && exec ../venv/bin/uvicorn oddly.server:app --port 8000
