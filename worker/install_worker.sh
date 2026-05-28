#!/usr/bin/env bash
# DistGPU worker: создание .venv и установка зависимостей (Linux / macOS).
#
# Переменные окружения:
#   PYTHON          — какой интерпретатор использовать (по умолчанию: python3 или python)
#   DISTGPU_TORCH   — пресет PyTorch (см. ниже). По умолчанию: на macOS — mps, иначе cu124
#   DISTGPU_PYTORCH_INDEX_URL — если задан, игнорирует пресет и ставит torch так:
#                     pip install torch --index-url "$DISTGPU_PYTORCH_INDEX_URL"
#   DISTGPU_PYTORCH_EXTRA  — доп. аргументы к pip при установке torch (например --pre)
#
# Пресеты DISTGPU_TORCH (колёса с download.pytorch.org, кроме cpu/mps):
#   cpu           — только CPU
#   mps           — macOS, torch с PyPI (MPS при поддержке)
#   cu118, cu121, cu124, cu126, cu128 — стабильные сборки CUDA (см. актуальность на pytorch.org)
#   nightly-cu128 — ночные сборки под CUDA 12.8 (часто нужны для RTX 50xx / Blackwell)
#
# Примеры:
#   ./install_worker.sh
#   DISTGPU_TORCH=nightly-cu128 ./install_worker.sh
#   DISTGPU_PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ./install_worker.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MIN_PY="3.10"

pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "$PYTHON"
    return
  fi
  if command -v python3.12 >/dev/null 2>&1; then echo "python3.12"; return; fi
  if command -v python3.11 >/dev/null 2>&1; then echo "python3.11"; return; fi
  if command -v python3.10 >/dev/null 2>&1; then echo "python3.10"; return; fi
  if command -v python3 >/dev/null 2>&1; then echo "python3"; return; fi
  if command -v python >/dev/null 2>&1; then echo "python"; return; fi
  echo ""
}

PY="$(pick_python)"
if [[ -z "$PY" ]]; then
  echo "[install] Не найден Python. Установите Python ${MIN_PY}+ или задайте PYTHON=/path/to/python" >&2
  exit 1
fi

if ! "$PY" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
  echo "[install] Нужен Python ${MIN_PY}+. Сейчас: $($PY -V)" >&2
  exit 1
fi

echo "[install] Используется: $($PY -V) ($PY)"

VENV="$SCRIPT_DIR/.venv"
if [[ ! -d "$VENV" ]]; then
  echo "[install] Создаю виртуальное окружение: $VENV"
  "$PY" -m venv "$VENV"
else
  echo "[install] Виртуальное окружение уже есть: $VENV"
fi

PIP="$VENV/bin/pip"
PYV="$VENV/bin/python"

"$PYV" -m pip install -U pip wheel setuptools
"$PIP" install -r "$SCRIPT_DIR/requirements-base.txt"

install_torch() {
  local preset="${1:-cu124}"
  if [[ -n "${DISTGPU_PYTORCH_INDEX_URL:-}" ]]; then
    echo "[install] torch: пользовательский index-url"
    if [[ -n "${DISTGPU_PYTORCH_EXTRA:-}" ]]; then
      # shellcheck disable=SC2086
      "$PIP" install $DISTGPU_PYTORCH_EXTRA torch --index-url "$DISTGPU_PYTORCH_INDEX_URL"
    else
      "$PIP" install torch --index-url "$DISTGPU_PYTORCH_INDEX_URL"
    fi
    return
  fi

  case "$preset" in
    cpu)
      echo "[install] torch: CPU (PyPI)"
      "$PIP" install torch
      ;;
    mps|macos)
      echo "[install] torch: macOS / MPS (PyPI)"
      "$PIP" install torch
      ;;
    cu118)
      echo "[install] torch: CUDA 11.8 wheels"
      "$PIP" install torch --index-url https://download.pytorch.org/whl/cu118
      ;;
    cu121)
      echo "[install] torch: CUDA 12.1 wheels"
      "$PIP" install torch --index-url https://download.pytorch.org/whl/cu121
      ;;
    cu124)
      echo "[install] torch: CUDA 12.4 wheels"
      "$PIP" install torch --index-url https://download.pytorch.org/whl/cu124
      ;;
    cu126)
      echo "[install] torch: CUDA 12.6 wheels"
      "$PIP" install torch --index-url https://download.pytorch.org/whl/cu126
      ;;
    cu128)
      echo "[install] torch: CUDA 12.8 wheels (стабильный индекс; при ошибке попробуйте nightly-cu128)"
      "$PIP" install torch --index-url https://download.pytorch.org/whl/cu128
      ;;
    nightly-cu128)
      echo "[install] torch: nightly CUDA 12.8 (RTX 50 / Blackwell и новее — часто требуется именно он)"
      "$PIP" install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
      ;;
    *)
      echo "[install] Неизвестный DISTGPU_TORCH=$preset" >&2
      echo "[install] Допустимо: cpu mps cu118 cu121 cu124 cu126 cu128 nightly-cu128" >&2
      exit 1
      ;;
  esac
}

UNAME="$(uname -s)"
DEFAULT_PRESET="cu124"
if [[ "$UNAME" == "Darwin" ]]; then
  DEFAULT_PRESET="mps"
fi

TORCH_PRESET="${DISTGPU_TORCH:-$DEFAULT_PRESET}"
echo "[install] Пресет PyTorch: $TORCH_PRESET (смените через DISTGPU_TORCH=...)"
install_torch "$TORCH_PRESET"

echo "[install] Проверка импортов..."
"$PYV" "$SCRIPT_DIR/verify_env.py"

echo
echo "[install] Готово. Запуск воркера: ./run_worker.sh"
echo "[install] Перед запуском задайте SERVER_URL и WORKER_TOKEN при необходимости."
