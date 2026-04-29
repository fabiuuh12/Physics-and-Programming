#!/usr/bin/env python3
"""Run one cislunar navigation experiment."""

from __future__ import annotations

import argparse

from cislunar_nav.config import load_config
from cislunar_nav.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    rows = run_experiment(load_config(args.config))
    for row in rows:
        print(
            f"{row['filter']}: rms_pos={row['rms_position_error']:.3e}, "
            f"max_pos={row['max_position_error']:.3e}, max_risk={row['max_risk']:.3e}"
        )


if __name__ == "__main__":
    main()
