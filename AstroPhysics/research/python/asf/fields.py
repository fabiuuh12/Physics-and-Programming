"""ASF field construction and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .numerics import gradient, smooth_box


@dataclass
class FieldState:
    rho: np.ndarray
    pressure: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    temperature: np.ndarray
    structure: np.ndarray
    phi_n: np.ndarray | None = None
    phi_eff: np.ndarray | None = None


def build_controlled_fields(config: dict[str, Any], structured: bool = True) -> FieldState:
    """Build deterministic density and baryonic-structure fields."""
    grid = config["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    rng = np.random.default_rng(int(config.get("seed", 0)))

    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y)
    r2 = xx * xx + yy * yy

    rho = 0.2 + 2.0 * np.exp(-r2 / 0.12) + 0.45 * np.exp(-((xx - 0.28) ** 2 + (yy + 0.18) ** 2) / 0.035)
    pressure = 0.8 + 0.3 * np.exp(-r2 / 0.25)
    temperature = 1.0 + 0.2 * np.exp(-((xx + 0.18) ** 2 + (yy - 0.22) ** 2) / 0.08)

    if structured:
        phase = 3.0 * np.pi * xx + 2.0 * np.pi * yy
        bx = 0.10 + 0.08 * np.sin(phase)
        by = 0.08 * np.cos(2.0 * np.pi * xx)
        vx = -yy * np.exp(-r2 / 0.35) + 0.03 * rng.normal(size=(ny, nx))
        vy = xx * np.exp(-r2 / 0.35) + 0.03 * rng.normal(size=(ny, nx))
        temperature = temperature + 0.12 * np.sin(4.0 * np.pi * xx) * np.cos(3.0 * np.pi * yy)
    else:
        bx = np.full((ny, nx), 0.1)
        by = np.zeros((ny, nx))
        vx = np.zeros((ny, nx))
        vy = np.zeros((ny, nx))

    structure = compute_structure_field(rho, pressure, bx, by, vx, vy, temperature, config)
    return FieldState(rho, pressure, bx, by, vx, vy, temperature, structure)


def compute_structure_field(
    rho: np.ndarray,
    pressure: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    temperature: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Compute the dimensionless ASF field S_A."""
    physics = config["physics"]
    grid = config["grid"]
    dx = float(grid["dx"])
    dy = float(grid["dy"])

    rho_floor = float(physics.get("rho_floor", 1.0e-12))
    pressure_floor = float(physics.get("pressure_floor", 1.0e-12))
    rho0 = float(physics["rho0"])
    mu0 = float(physics["mu0"])
    omega0 = float(physics["omega0"])
    temperature0 = float(physics["temperature0"])
    length0 = float(physics["length0"])

    safe_rho = np.maximum(rho, rho_floor)
    safe_pressure = np.maximum(pressure, pressure_floor)
    density_term = np.log(safe_rho / rho0)

    magnetic_term = (bx * bx + by * by) / (2.0 * mu0 * safe_pressure)

    dvx_dy = np.gradient(vx, dy, axis=0, edge_order=2)
    dvy_dx = np.gradient(vy, dx, axis=1, edge_order=2)
    vorticity = dvy_dx - dvx_dy
    vorticity_term = (vorticity * vorticity) / (omega0 * omega0)

    grad_tx, grad_ty = gradient(temperature, dx, dy)
    thermal_term = (length0 * length0 * (grad_tx * grad_tx + grad_ty * grad_ty)) / (temperature0 * temperature0)

    structure = (
        density_term
        + float(physics["chi_B"]) * magnetic_term
        + float(physics["chi_omega"]) * vorticity_term
        + float(physics["chi_T"]) * thermal_term
    )
    return smooth_box(structure, int(config.get("solver", {}).get("smooth_passes", 0)))


def effective_potential(phi_n: np.ndarray, structure: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Compute Phi_eff = Phi_N + lambda_A S_A."""
    return phi_n + float(config["physics"]["lambda_A"]) * structure


def acceleration(phi: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return acceleration components from a potential."""
    grad_x, grad_y = gradient(phi, dx, dy)
    return -grad_x, -grad_y
