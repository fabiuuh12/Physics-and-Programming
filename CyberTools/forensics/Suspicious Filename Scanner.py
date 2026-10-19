"""
Suspicious Filename Scanner

Flags filenames with double extensions or risky patterns.
Usage:
    python "Suspicious Filename Scanner.py" --path C:\path\to\scan
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DOUBLE_EXT_RE = re.compile(r"\.(pdf|doc|docx|txt|jpg|png|xls|xlsx)\.(exe|js|vbs|bat|cmd|scr)$", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan for suspicious filename patterns")
    parser.add_argument("--path", required=True, help="Directory to scan")
    args = parser.parse_args()

    base = Path(args.path)
    if not base.exists() or not base.is_dir():
        print(f"Invalid directory: {base}")
        return 1

    hits = []
    for path in base.rglob("*"):
        if path.is_file():
            name = path.name
            if DOUBLE_EXT_RE.search(name) or name.lower().endswith(".lnk"):
                hits.append(path)

    if not hits:
        print("No suspicious filenames found.")
        return 0

    print("Suspicious filenames:")
    for path in hits:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
