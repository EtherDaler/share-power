#!/usr/bin/env bash
# Запуск DistGPU worker из виртуального окружения (Linux / macOS).
#
# Переменные (пример):
#   export SERVER_URL=ws://127.0.0.1:8765/worker
#   export WORKER_TOKEN=secret-token-123
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$SCRIPT_DIR/worker.py"
PY="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -f "$WORKER" ]]; then
  echo "[run] Не найден $WORKER" >&2
  exit 1
fi

if [[ -x "$PY" ]]; then
  exec "$PY" "$WORKER" "$@"
fi

BASE="${PYTHON:-python3}"
exec "$BASE" "$WORKER" "$@"
