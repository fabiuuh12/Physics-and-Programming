"""
Certificate Expiry Checker

Checks TLS certificate expiry dates for a list of hostnames.
Usage:
    python "Certificate Expiry Checker.py" --file hosts.txt
"""

from __future__ import annotations

import argparse
import socket
import ssl
from datetime import datetime
from pathlib import Path


def get_cert_expiry(host: str, port: int = 443) -> str:
    context = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=5) as sock:
        with context.wrap_socket(sock, server_hostname=host) as ssock:
            cert = ssock.getpeercert()
    not_after = cert.get("notAfter")
    if not not_after:
        return "unknown"
    dt = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
    days = (dt - datetime.utcnow()).days
    return f"{dt.isoformat()}Z ({days} days)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check TLS certificate expiry")
    parser.add_argument("--file", required=True, help="File with hostnames")
    parser.add_argument("--port", type=int, default=443)
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    hosts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not hosts:
        print("No hosts found.")
        return 1

    for host in hosts:
        try:
            expiry = get_cert_expiry(host, args.port)
            print(f"{host:40}  {expiry}")
        except Exception as exc:
            print(f"{host:40}  ERROR: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
