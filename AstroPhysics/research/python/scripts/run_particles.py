#!/usr/bin/env python3
"""Run ASF test-particle orbit comparisons."""

from __future__ import annotations

import argparse
import math

import numpy as np

from asf.config import load_config, output_directory
from asf.diagnostics import plot_field_grid, save_npz, write_json
from asf.fields import build_controlled_fields, effective_potential
from asf.numerics import poisson_jacobi
from asf.particles import initialize_ring_particles, integrate_particles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    out = output_directory(config)
    grid = config["grid"]
    solver = config["solver"]
    pconf = config["particles"]
    dx = float(grid["dx"])
    dy = float(grid["dy"])

    state = build_controlled_fields(config, structured=True)
    source = 4.0 * math.pi * float(config["physics"]["G"]) * state.rho
    phi_n, residuals = poisson_jacobi(source, dx, dy, int(solver["iterations"]), float(solver["tolerance"]))
    phi_eff = effective_potential(phi_n, state.structure, config)

    initial = initialize_ring_particles(config)
    hist_n = integrate_particles(initial, phi_n, dx, dy, float(pconf["dt"]), int(pconf["steps"]))
    hist_eff = integrate_particles(initial, phi_eff, dx, dy, float(pconf["dt"]), int(pconf["steps"]))
    separation = np.sqrt((hist_eff[:, :, 0] - hist_n[:, :, 0]) ** 2 + (hist_eff[:, :, 1] - hist_n[:, :, 1]) ** 2)

    save_npz(out / "particles.npz", phi_n=phi_n, phi_eff=phi_eff, structure=state.structure, hist_n=hist_n, hist_eff=hist_eff)
    write_json(
        out / "summary.json",
        {
            "experiment_name": config["experiment_name"],
            "poisson_final_residual": residuals[-1] if residuals else None,
            "mean_trajectory_separation": float(separation.mean()),
            "max_trajectory_separation": float(separation.max()),
            "particle_count": int(pconf["count"]),
            "steps": int(pconf["steps"]),
        },
    )
    plot_field_grid(out / "orbit_fields.png", {"S_A": state.structure, "Phi_N": phi_n, "Phi_eff": phi_eff})
    print(f"Wrote particle outputs to {out}")


if __name__ == "__main__":
    main()
