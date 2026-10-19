"""JSON array to CSV converter."""
import csv
import json
from pathlib import Path


def flatten(prefix: str, obj, row: dict[str, str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, row)
    elif isinstance(obj, list):
        row[prefix] = json.dumps(obj, ensure_ascii=False)
    else:
        row[prefix] = "" if obj is None else str(obj)


def main() -> None:
    src = Path(input("Input JSON: ").strip())
    dst = Path(input("Output CSV: ").strip())
    if not src.is_file():
        print("Input not found")
        return

    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("Expected JSON array")
        return

    rows = []
    headers = set()
    for item in data:
        row: dict[str, str] = {}
        flatten("", item, row)
        rows.append(row)
        headers.update(row.keys())

    cols = sorted(headers)
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows")


if __name__ == "__main__":
    main()

