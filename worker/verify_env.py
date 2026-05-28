"""Проверка окружения воркера после установки (вызывается install-скриптами)."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import websockets  # noqa: F401
    except ImportError:
        print("[verify] Ошибка: не установлен пакет websockets", file=sys.stderr)
        return 1
    try:
        import torch
    except ImportError:
        print("[verify] Ошибка: не установлен torch", file=sys.stderr)
        return 1

    print("[verify] websockets OK")
    print(f"[verify] torch {torch.__version__}")
    print(f"[verify] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"[verify] GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"[verify] GPU: (не удалось прочитать имя: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
