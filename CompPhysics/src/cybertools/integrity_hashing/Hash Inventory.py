"""
Hash Inventory Tool

Computes SHA256 hashes for files in a directory.
Usage:
    python "Hash Inventory.py" --path C:\path\to\folder --out hashes.csv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute SHA256 hashes for files")
    parser.add_argument("--path", required=True, help="Directory to hash")
    parser.add_argument("--out", required=False, help="Output CSV file")
    args = parser.parse_args()

    base = Path(args.path)
    if not base.exists() or not base.is_dir():
        print(f"Invalid directory: {base}")
        return 1

    rows = ["path,sha256"]
    for path in base.rglob("*"):
        if path.is_file():
            try:
                digest = sha256_file(path)
                rows.append(f"{path},{digest}")
            except Exception as exc:
                rows.append(f"{path},ERROR:{exc}")

    output = "\n".join(rows)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote: {args.out}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
