#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "Virtualenv nao encontrado. Crie com: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

# agent-mini uses the workspace from config.json, not the process cwd. Default
# it to this project so the agent does not fall back to ~/.agent-mini/workspace.
# Override with AGENT_MINI_WORKSPACE=/path (or chat --workspace /path).
if [ -z "${AGENT_MINI_WORKSPACE:-}" ]; then
    export AGENT_MINI_WORKSPACE="$SCRIPT_DIR"
fi

cd "$SCRIPT_DIR"
exec "$PYTHON" -m src.main "$@"
