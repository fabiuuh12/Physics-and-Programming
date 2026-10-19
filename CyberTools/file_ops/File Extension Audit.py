"""
File Extension Audit

Scans a directory and reports potentially risky file types.
Usage:
    python "File Extension Audit.py" --path C:\path\to\scan
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

RISKY_EXTENSIONS = {
    ".exe", ".dll", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".jar",
    ".scr", ".msi", ".reg", ".lnk", ".hta", ".wsf",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit file extensions in a directory")
    parser.add_argument("--path", required=True, help="Directory to scan")
    args = parser.parse_args()

    base = Path(args.path)
    if not base.exists() or not base.is_dir():
        print(f"Invalid directory: {base}")
        return 1

    ext_counts = Counter()
    risky_hits = []

    for path in base.rglob("*"):
        if path.is_file():
            ext = path.suffix.lower()
            ext_counts[ext] += 1
            if ext in RISKY_EXTENSIONS:
                risky_hits.append(path)

    print("Top extensions:")
    for ext, count in ext_counts.most_common(15):
        ext_label = ext if ext else "<no extension>"
        print(f"  {ext_label:>14} : {count}")

    print("\nPotentially risky files:")
    if not risky_hits:
        print("  None found.")
    else:
        for path in risky_hits[:200]:
            print(f"  {path}")
        if len(risky_hits) > 200:
            print(f"  ... and {len(risky_hits) - 200} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
