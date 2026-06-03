"""Decode JWT header/payload without signature verification."""
import base64
import json


def b64url_decode(seg: str) -> bytes:
    seg += "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg.encode("ascii"))


def main() -> None:
    token = input("JWT: ").strip()
    parts = token.split(".")
    if len(parts) != 3:
        print("Invalid JWT format")
        return

    try:
        header = json.loads(b64url_decode(parts[0]).decode("utf-8", errors="ignore"))
        payload = json.loads(b64url_decode(parts[1]).decode("utf-8", errors="ignore"))
    except Exception as e:
        print(f"Decode failed: {e}")
        return

    print("\nHeader:")
    print(json.dumps(header, indent=2))
    print("\nPayload:")
    print(json.dumps(payload, indent=2))
    print("\nSignature segment:")
    print(parts[2])


if __name__ == "__main__":
    main()

