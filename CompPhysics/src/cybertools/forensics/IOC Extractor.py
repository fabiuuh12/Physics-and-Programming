"""Extract common IOCs from text."""
import re
from pathlib import Path


PATTERNS = {
    "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "domain": r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b",
    "url": r"\bhttps?://[^\s'\"<>]+",
    "email": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
    "sha256": r"\b[a-fA-F0-9]{64}\b",
    "md5": r"\b[a-fA-F0-9]{32}\b",
}


def main() -> None:
    path = Path(input("Text file: ").strip())
    if not path.is_file():
        print("File not found")
        return

    text = path.read_text(encoding="utf-8", errors="ignore")
    for name, pat in PATTERNS.items():
        vals = sorted(set(re.findall(pat, text)))
        print(f"\n{name.upper()} ({len(vals)}):")
        for v in vals[:300]:
            print(f"- {v}")


if __name__ == "__main__":
    main()

