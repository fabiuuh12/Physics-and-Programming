#!/usr/bin/env python3
"""Plot diagnostics from a generated ASF field file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from asf.diagnostics import radial_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.fields)
    radius, phi_n = radial_profile(data["phi_n"], 1.0, 1.0)
    _, phi_eff = radial_profile(data["phi_eff"], 1.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(radius, phi_n, label="Phi_N")
    ax.plot(radius, phi_eff, label="Phi_eff")
    ax.set_xlabel("radius [grid units]")
    ax.set_ylabel("radial mean potential")
    ax.legend()
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Wrote diagnostics plot to {output}")


if __name__ == "__main__":
    main()
