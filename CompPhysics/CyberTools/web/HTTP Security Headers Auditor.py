"""Check common HTTP security headers."""
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def normalize(url: str) -> str:
    return url if urlparse(url).scheme else f"https://{url}"


def main() -> None:
    url = normalize(input("URL: ").strip())
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=12) as r:
            headers = {k.lower(): v for k, v in r.headers.items()}
            print(f"Status: {r.status}")
    except Exception as e:
        print(f"Request failed: {e}")
        return

    checks = [
        "strict-transport-security",
        "content-security-policy",
        "x-content-type-options",
        "x-frame-options",
        "referrer-policy",
        "permissions-policy",
    ]

    print()
    for c in checks:
        v = headers.get(c)
        print(f"[OK] {c}: {v}" if v else f"[MISSING] {c}")


if __name__ == "__main__":
    main()

