"""Flag files with suspicious ransomware-like extensions."""
from pathlib import Path


SUSPICIOUS = {
    ".locky",
    ".crypt",
    ".crypted",
    ".encrypted",
    ".enc",
    ".crypto",
    ".zepto",
    ".cerber",
    ".wncry",
    ".wannacry",
    ".ryuk",
    ".conti",
    ".akira",
}


def main() -> None:
    root = Path(input("Folder to scan: ").strip())
    if not root.is_dir():
        print("Invalid folder")
        return

    hits = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUSPICIOUS]
    print(f"Potential hits: {len(hits)}")
    for h in hits[:1000]:
        print(f"- {h}")


if __name__ == "__main__":
    main()

