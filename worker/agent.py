"""Совместимость: используйте worker.py (один файл для распространения)."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("worker.py")), run_name="__main__")
