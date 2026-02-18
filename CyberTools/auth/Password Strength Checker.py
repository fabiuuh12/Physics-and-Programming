"""
Password Strength Checker

Checks basic password strength heuristics and estimates entropy.
Usage:
    python "Password Strength Checker.py" --password "yourpassword"
    python "Password Strength Checker.py" --file passwords.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def estimate_entropy(password: str) -> float:
    if not password:
        return 0.0

    pool = 0
    if any(c.islower() for c in password):
        pool += 26
    if any(c.isupper() for c in password):
        pool += 26
    if any(c.isdigit() for c in password):
        pool += 10
    if any(not c.isalnum() for c in password):
        pool += 33

    return len(password) * math.log2(pool or 1)


def score_password(password: str) -> dict:
    issues = []

    if len(password) < 10:
        issues.append("Too short (< 10 characters)")
    if password.lower() == password or password.upper() == password:
        issues.append("Use mixed case")
    if not any(c.isdigit() for c in password):
        issues.append("Add digits")
    if not any(not c.isalnum() for c in password):
        issues.append("Add symbols")
    if len(set(password)) < max(6, len(password) // 3):
        issues.append("Low character variety")

    entropy = estimate_entropy(password)
    if entropy < 40:
        strength = "Weak"
    elif entropy < 60:
        strength = "Moderate"
    elif entropy < 80:
        strength = "Strong"
    else:
        strength = "Very Strong"

    return {
        "password": password,
        "length": len(password),
        "entropy_bits": round(entropy, 2),
        "strength": strength,
        "issues": issues,
    }


def analyze_passwords(passwords: list[str]) -> None:
    for pw in passwords:
        result = score_password(pw)
        print("-" * 60)
        print(f"Password: {'*' * len(pw)}")
        print(f"Length: {result['length']}")
        print(f"Estimated entropy: {result['entropy_bits']} bits")
        print(f"Strength: {result['strength']}")
        if result["issues"]:
            print("Issues:")
            for issue in result["issues"]:
                print(f"  - {issue}")
        else:
            print("No obvious issues detected.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Password strength checker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--password", help="Single password to evaluate")
    group.add_argument("--file", help="File with one password per line")

    args = parser.parse_args()

    if args.password is not None:
        analyze_passwords([args.password.strip()])
        return 0

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    passwords = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not passwords:
        print("No passwords found in file.")
        return 1

    analyze_passwords(passwords)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
