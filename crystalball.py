"""Bootstrap runner for the CrystalBall project.

Run with: python crystalball.py

This script adds the repository root to sys.path so `from src import ...` imports
work both when running as a script and when using the `-m` module style.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.main import main
except Exception:
    # last-resort: add src to path and retry
    src_path = os.path.join(ROOT, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from src.main import main

if __name__ == '__main__':
    main()
