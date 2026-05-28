#!/usr/bin/env bash
# Сборка нативного воркера distgpu-worker (Linux / macOS).
# Требования: CMake 3.16+, компилятор C++17, python в PATH (для обучения), nvidia-smi (опционально).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[install_native] Не найдено: $1" >&2
    exit 1
  }
}

need_cmd cmake
need_cmd python3 || need_cmd python

if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi
echo "[install_native] Обнаружен интерпретатор: $($PY -V 2>&1)"

CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"

echo "[install_native] Конфигурация CMake → $BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -DIXWEBSOCKET_USE_TLS=OFF -DIXWEBSOCKET_USE_ZLIB=OFF

echo "[install_native] Сборка…"
cmake --build "$BUILD_DIR" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

BIN=""
if [[ -f "$BUILD_DIR/distgpu-worker" ]]; then
  BIN="$BUILD_DIR/distgpu-worker"
elif [[ -f "$BUILD_DIR/Release/distgpu-worker.exe" ]]; then
  BIN="$BUILD_DIR/Release/distgpu-worker.exe"
elif [[ -f "$BUILD_DIR/distgpu-worker.exe" ]]; then
  BIN="$BUILD_DIR/distgpu-worker.exe"
else
  echo "[install_native] Не найден бинарник после сборки (искал в $BUILD_DIR)" >&2
  exit 1
fi

if [[ "$BIN" == *.exe ]]; then
  OUT="$ROOT/distgpu-worker.exe"
else
  OUT="$ROOT/distgpu-worker"
fi
cp -f "$BIN" "$OUT"
chmod +x "$OUT" 2>/dev/null || true

echo "[install_native] Готово. Запуск:"
echo "  export SERVER_URL=ws://127.0.0.1:8765/worker"
echo "  export WORKER_TOKEN=..."
echo "  $OUT"
