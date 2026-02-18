"""
DNS Resolver Check

Resolves domains to A/AAAA records and reports results.
Usage:
    python "DNS Resolver Check.py" --file domains.txt
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path


def resolve_domain(domain: str) -> list[str]:
    ips = set()
    try:
        for res in socket.getaddrinfo(domain, None):
            ips.add(res[4][0])
    except Exception:
        return []
    return sorted(ips)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve domains to IPs")
    parser.add_argument("--file", required=True, help="File with domains")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    domains = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not domains:
        print("No domains found.")
        return 1

    for domain in domains:
        ips = resolve_domain(domain)
        if ips:
            print(f"{domain:35}  {', '.join(ips)}")
        else:
            print(f"{domain:35}  (no result)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
