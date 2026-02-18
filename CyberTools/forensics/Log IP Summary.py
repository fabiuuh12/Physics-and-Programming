"""
Log IP Summary

Parses a log file for IPv4/IPv6 addresses and summarizes counts.
Usage:
    python "Log IP Summary.py" --log access.log --top 20
"""

from __future__ import annotations

import argparse
import ipaddress
import re
from collections import Counter
from pathlib import Path

IP_REGEX = re.compile(
    r"(?:(?:\d{1,3}\.){3}\d{1,3})|(?:[a-fA-F0-9:]+:+[a-fA-F0-9]+)")


def classify_ip(ip_str: str) -> str:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return "invalid"
    if ip.is_private:
        return "private"
    if ip.is_loopback:
        return "loopback"
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved:
        return "reserved"
    return "public"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize IPs in a log file")
    parser.add_argument("--log", required=True, help="Path to log file")
    parser.add_argument("--top", type=int, default=20, help="Top N IPs to show")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"Log file not found: {path}")
        return 1

    data = path.read_text(encoding="utf-8", errors="ignore")
    ips = IP_REGEX.findall(data)
    counts = Counter(ips)

    print(f"Total IPs found: {len(ips)}")
    print("\nTop IPs:")
    for ip, count in counts.most_common(args.top):
        print(f"{ip:>40}  {count:>6}  ({classify_ip(ip)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
