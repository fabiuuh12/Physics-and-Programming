"""
Windows Login Failures Summary

Parses Windows Security Event Log (exported) for failed logons (Event ID 4625).
Usage:
    python "Windows Login Failures Summary.py" --log security.evtx

Note: Use Windows Event Viewer to export Security log to EVTX.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

try:
    import Evtx.Evtx as evtx
except ImportError:
    evtx = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Windows failed logons (Event ID 4625)")
    parser.add_argument("--log", required=True, help="Path to .evtx file")
    args = parser.parse_args()

    if evtx is None:
        print("Missing dependency: python-evtx (pip install python-evtx)")
        return 1

    path = Path(args.log)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    ip_counter = Counter()
    user_counter = Counter()

    with evtx.Evtx(str(path)) as log:
        for record in log.records():
            xml = record.xml()
            if "<EventID>4625</EventID>" not in xml:
                continue
            # Rough extraction from XML text (heuristic parsing)
            if "IpAddress" in xml:
                for line in xml.splitlines():
                    if "IpAddress" in line:
                        val = line.split(">")[-1].split("<")[0].strip()
                        if val and val != "-":
                            ip_counter[val] += 1
            if "TargetUserName" in xml:
                for line in xml.splitlines():
                    if "TargetUserName" in line:
                        val = line.split(">")[-1].split("<")[0].strip()
                        if val and val != "-":
                            user_counter[val] += 1

    print("Top IPs:")
    for ip, count in ip_counter.most_common(20):
        print(f"  {ip:>24}  {count}")

    print("\nTop Usernames:")
    for user, count in user_counter.most_common(20):
        print(f"  {user:>24}  {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
