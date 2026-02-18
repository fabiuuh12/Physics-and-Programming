"""
URL Heuristic Scanner

Scans URLs from a file and flags suspicious patterns (heuristic only).
Usage:
    python "URL Heuristic Scanner.py" --file urls.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {"zip", "mov", "top", "xyz", "click", "beauty", "country", "gq", "work"}


def score_url(url: str) -> tuple[int, list[str]]:
    reasons = []
    score = 0

    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.netloc.lower()

    if host.count("-") >= 3:
        score += 1
        reasons.append("many hyphens in hostname")

    if re.search(r"\d{4,}", host):
        score += 1
        reasons.append("long digit sequence in hostname")

    parts = host.split(".")
    if parts and parts[-1] in SUSPICIOUS_TLDS:
        score += 1
        reasons.append("suspicious TLD")

    if "@" in url:
        score += 1
        reasons.append("contains @ in URL")

    if len(host) > 40:
        score += 1
        reasons.append("very long hostname")

    if parsed.scheme not in {"http", "https"}:
        score += 1
        reasons.append("non-http/https scheme")

    return score, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Heuristic URL scanner")
    parser.add_argument("--file", required=True, help="File with URLs (one per line)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    urls = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not urls:
        print("No URLs found.")
        return 1

    for url in urls:
        score, reasons = score_url(url)
        flag = "FLAG" if score >= 2 else "OK"
        print(f"[{flag}] {url}")
        if reasons:
            for r in reasons:
                print(f"  - {r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
