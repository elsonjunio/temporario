#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "Virtualenv nao encontrado. Crie com: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
exec "$PYTHON" -m src.main "$@"
