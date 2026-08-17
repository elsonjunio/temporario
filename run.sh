#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    echo "Virtualenv nao encontrado em $SCRIPT_DIR/.venv" >&2
    exit 1
fi

export VIRTUAL_ENV="$SCRIPT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

exec "$VIRTUAL_ENV/bin/python" "$SCRIPT_DIR/src/bike.py"
