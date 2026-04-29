from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from asf.config import load_config
from asf.fields import acceleration, build_controlled_fields, effective_potential
from asf.numerics import gradient, laplacian, poisson_jacobi, poisson_residual


CONFIG = Path("AstroPhysics/research/experiments/baseline_static_field/config.json")


def test_structure_field_is_finite() -> None:
    config = load_config(CONFIG)
    state = build_controlled_fields(config, structured=True)
    assert np.all(np.isfinite(state.structure))
    assert state.structure.shape == state.rho.shape


def test_uniform_structure_has_zero_extra_acceleration() -> None:
    config = load_config(CONFIG)
    phi_n = np.zeros((16, 16), dtype=float)
    uniform_structure = np.full((16, 16), 3.2)
    phi_eff = effective_potential(phi_n, uniform_structure, config)
    ax, ay = acceleration(phi_eff, 1.0, 1.0)
    assert np.max(np.abs(ax)) < 1.0e-12
    assert np.max(np.abs(ay)) < 1.0e-12


def test_lambda_zero_recovers_newtonian_potential() -> None:
    config = load_config(CONFIG)
    config["physics"]["lambda_A"] = 0.0
    phi_n = np.arange(25, dtype=float).reshape(5, 5)
    structure = np.ones((5, 5), dtype=float) * 9.0
    assert np.allclose(effective_potential(phi_n, structure, config), phi_n)


def test_gradient_and_laplacian_on_quadratic_field() -> None:
    n = 32
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    xx, yy = np.meshgrid(x, y)
    field = xx * xx + yy * yy
    gx, gy = gradient(field, dx, dy)
    lap = laplacian(field, dx, dy)
    assert np.allclose(gx[2:-2, 2:-2], 2.0 * xx[2:-2, 2:-2], atol=1.0e-10)
    assert np.allclose(gy[2:-2, 2:-2], 2.0 * yy[2:-2, 2:-2], atol=1.0e-10)
    assert np.allclose(lap[2:-2, 2:-2], 4.0, atol=1.0e-10)


def test_poisson_residual_decreases() -> None:
    n = 32
    source = np.zeros((n, n), dtype=float)
    source[n // 2, n // 2] = 1.0
    initial = poisson_residual(np.zeros_like(source), source, 1.0, 1.0)
    phi, residuals = poisson_jacobi(source, 1.0, 1.0, iterations=120, tolerance=1.0e-12)
    assert residuals
    assert math.isfinite(residuals[-1])
    assert residuals[-1] < initial
    assert np.all(np.isfinite(phi))
