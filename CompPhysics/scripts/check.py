#!/usr/bin/env python3
"""Run the test suite without creating bytecode caches beside project code."""
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

try:
    import pytest
except ImportError:
    print('Install test dependencies: python -m pip install -r CompPhysics/config/requirements-dev.txt', file=sys.stderr)
    raise SystemExit(1)

if __name__ == '__main__':
    raise SystemExit(pytest.main(sys.argv[1:] or ['-q']))
