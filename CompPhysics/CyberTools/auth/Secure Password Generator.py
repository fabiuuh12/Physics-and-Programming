"""
Secure Password Generator

Generates random passwords with configurable length and character sets.
Usage:
    python "Secure Password Generator.py" --length 20 --count 5
"""

from __future__ import annotations

import argparse
import secrets
import string


def build_charset(use_upper: bool, use_lower: bool, use_digits: bool, use_symbols: bool) -> str:
    chars = ""
    if use_upper:
        chars += string.ascii_uppercase
    if use_lower:
        chars += string.ascii_lowercase
    if use_digits:
        chars += string.digits
    if use_symbols:
        chars += "!@#$%^&*()-_=+[]{};:,.?/"
    return chars


def main() -> int:
    parser = argparse.ArgumentParser(description="Secure password generator")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--no-upper", action="store_true")
    parser.add_argument("--no-lower", action="store_true")
    parser.add_argument("--no-digits", action="store_true")
    parser.add_argument("--no-symbols", action="store_true")
    args = parser.parse_args()

    charset = build_charset(
        use_upper=not args.no_upper,
        use_lower=not args.no_lower,
        use_digits=not args.no_digits,
        use_symbols=not args.no_symbols,
    )

    if not charset:
        print("Error: No character sets selected.")
        return 1

    for _ in range(args.count):
        pw = "".join(secrets.choice(charset) for _ in range(args.length))
        print(pw)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
