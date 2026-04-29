#!/usr/bin/env python3
"""Run the baseline ASF static-field experiment."""

from __future__ import annotations

import argparse
import math

from asf.config import load_config, output_directory
from asf.diagnostics import plot_field_grid, save_npz, write_json
from asf.fields import acceleration, build_controlled_fields, effective_potential
from asf.numerics import poisson_jacobi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    out = output_directory(config)
    grid = config["grid"]
    solver = config["solver"]
    dx = float(grid["dx"])
    dy = float(grid["dy"])

    state = build_controlled_fields(config, structured=True)
    source = 4.0 * math.pi * float(config["physics"]["G"]) * state.rho
    phi_n, residuals = poisson_jacobi(source, dx, dy, int(solver["iterations"]), float(solver["tolerance"]))
    phi_eff = effective_potential(phi_n, state.structure, config)
    ax_n, ay_n = acceleration(phi_n, dx, dy)
    ax_eff, ay_eff = acceleration(phi_eff, dx, dy)
    delta_accel = ((ax_eff - ax_n) ** 2 + (ay_eff - ay_n) ** 2) ** 0.5

    save_npz(
        out / "fields.npz",
        rho=state.rho,
        pressure=state.pressure,
        temperature=state.temperature,
        structure=state.structure,
        phi_n=phi_n,
        phi_eff=phi_eff,
        delta_accel=delta_accel,
    )
    write_json(
        out / "summary.json",
        {
            "experiment_name": config["experiment_name"],
            "poisson_initial_residual": residuals[0] if residuals else None,
            "poisson_final_residual": residuals[-1] if residuals else None,
            "poisson_iterations": len(residuals),
            "structure_min": float(state.structure.min()),
            "structure_max": float(state.structure.max()),
            "delta_accel_mean": float(delta_accel.mean()),
            "delta_accel_max": float(delta_accel.max()),
        },
    )
    plot_field_grid(
        out / "field_grid.png",
        {
            "rho": state.rho,
            "S_A": state.structure,
            "Phi_N": phi_n,
            "Phi_eff": phi_eff,
            "|Delta a|": delta_accel,
        },
    )
    print(f"Wrote static-field outputs to {out}")


if __name__ == "__main__":
    main()
