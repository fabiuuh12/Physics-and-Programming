"""Directory snapshot + diff."""
import json
from pathlib import Path


def snapshot(root: Path) -> dict:
    out = {}
    for p in root.rglob("*"):
        if p.is_file():
            try:
                st = p.stat()
                out[str(p.relative_to(root))] = {"size": st.st_size, "mtime": st.st_mtime}
            except OSError:
                pass
    return out


def main() -> None:
    print("1) Create snapshot")
    print("2) Compare snapshot")
    choice = input("Select: ").strip()

    if choice == "1":
        root = Path(input("Folder: ").strip())
        out = Path(input("Output json: ").strip())
        if not root.is_dir():
            print("Invalid folder")
            return
        out.write_text(json.dumps({"root": str(root), "files": snapshot(root)}, indent=2), encoding="utf-8")
        print("Saved")
        return

    if choice == "2":
        snap_file = Path(input("Snapshot json: ").strip())
        root = Path(input("Current folder: ").strip())
        if not snap_file.is_file() or not root.is_dir():
            print("Invalid input")
            return

        old = json.loads(snap_file.read_text(encoding="utf-8")).get("files", {})
        new = snapshot(root)
        old_set, new_set = set(old), set(new)
        added = sorted(new_set - old_set)
        removed = sorted(old_set - new_set)
        changed = sorted(k for k in old_set & new_set if old[k] != new[k])

        print(f"Added: {len(added)}  Removed: {len(removed)}  Changed: {len(changed)}")
        for label, items in (("Added", added), ("Removed", removed), ("Changed", changed)):
            if items:
                print(f"\n{label}:")
                for i in items[:200]:
                    print(f"- {i}")
        return

    print("Invalid choice")


if __name__ == "__main__":
    main()

