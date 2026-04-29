#!/usr/bin/env python3
"""Run a cislunar navigation parameter scan."""

from __future__ import annotations

import argparse

from cislunar_nav.config import load_config
from cislunar_nav.experiments import run_parameter_scan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    rows = run_parameter_scan(load_config(args.config))
    print(f"Wrote {len(rows)} filter summaries")


if __name__ == "__main__":
    main()
