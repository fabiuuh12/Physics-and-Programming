"""Bulk File Renamer."""
from pathlib import Path


def main() -> None:
    folder = Path(input("Folder path: ").strip())
    if not folder.is_dir():
        print("Invalid folder")
        return

    old_txt = input("Replace text (optional): ")
    new_txt = input("With: ") if old_txt else ""
    prefix = input("Prefix (optional): ")
    suffix = input("Suffix before extension (optional): ")
    dry = input("Dry run? (y/n): ").strip().lower() != "n"

    changes = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        stem = p.stem.replace(old_txt, new_txt) if old_txt else p.stem
        new_p = p.with_name(f"{prefix}{stem}{suffix}{p.suffix}")
        if new_p != p:
            changes.append((p, new_p))

    if not changes:
        print("No changes")
        return

    for src, dst in changes:
        print(f"{src.name} -> {dst.name}")
    if dry:
        print("Dry run complete")
        return

    for src, dst in changes:
        if dst.exists():
            print(f"Skip (exists): {dst.name}")
            continue
        src.rename(dst)
    print("Done")


if __name__ == "__main__":
    main()
