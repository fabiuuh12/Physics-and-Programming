"""Subdomain finder using crt.sh certificate logs."""
import json
from urllib.parse import quote
from urllib.request import urlopen


def main() -> None:
    domain = input("Domain: ").strip().lower()
    if not domain:
        print("Domain required")
        return

    url = f"https://crt.sh/?q=%25.{quote(domain)}&output=json"
    try:
        with urlopen(url, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8", errors="ignore"))
    except Exception as e:
        print(f"Lookup failed: {e}")
        return

    names = set()
    for row in data:
        for n in str(row.get("name_value", "")).splitlines():
            n = n.strip().lower()
            if n.startswith("*."):
                n = n[2:]
            if n.endswith(domain):
                names.add(n)

    print(f"Found {len(names)} names")
    for n in sorted(names):
        print(f"- {n}")


if __name__ == "__main__":
    main()

