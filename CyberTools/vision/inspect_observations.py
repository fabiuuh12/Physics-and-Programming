#!/usr/bin/env python3
"""Print a compact summary of a Scene Observation Recorder database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


DEFAULT_DATABASE = Path(__file__).resolve().parent / "data" / "scene_observations.sqlite3"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", nargs="?", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    if not args.database.is_file():
        print(f"No observation database found at: {args.database}")
        return 1
    with sqlite3.connect(args.database) as connection:
        total = connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        sessions = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        print(f"Sessions: {sessions}\nObservation rows: {total}\n")
        for kind, label, count in connection.execute(
            """
            SELECT kind, label, COUNT(*)
            FROM observations
            GROUP BY kind, label
            ORDER BY kind, COUNT(*) DESC, label
            """
        ):
            print(f"{kind:8} {label:20} {count:6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
