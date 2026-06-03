"""Duplicate finder by SHA256."""
import hashlib
from collections import defaultdict
from pathlib import Path


CHUNK = 1024 * 1024


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    root = Path(input("Folder to scan: ").strip())
    if not root.is_dir():
        print("Invalid folder")
        return

    by_size: dict[int, list[Path]] = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file():
            try:
                by_size[p.stat().st_size].append(p)
            except OSError:
                pass

    groups: dict[str, list[Path]] = defaultdict(list)
    for files in by_size.values():
        if len(files) < 2:
            continue
        for f in files:
            try:
                groups[file_hash(f)].append(f)
            except OSError:
                pass

    dup_groups = [g for g in groups.values() if len(g) > 1]
    if not dup_groups:
        print("No duplicates found")
        return

    for i, g in enumerate(dup_groups, 1):
        print(f"\nGroup {i}:")
        for p in g:
            print(f"- {p}")


if __name__ == "__main__":
    main()

