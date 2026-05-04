"""Numerical helpers for Grover-search probability comparisons."""

from __future__ import annotations

import numpy as np


def grover_theta(database_size: int) -> float:
    """Return theta where sin(theta) = 1 / sqrt(database_size)."""
    if database_size <= 0:
        raise ValueError("database_size must be positive")
    return float(np.arcsin(1.0 / np.sqrt(database_size)))


def optimal_grover_iterations(database_size: int) -> int:
    """Return the standard nearest-integer Grover iteration estimate."""
    theta = grover_theta(database_size)
    return max(0, int(round((np.pi / (4.0 * theta)) - 0.5)))


def quantum_success_probs(database_size: int, iterations) -> np.ndarray:
    """Return target-state probabilities after each Grover iteration count."""
    theta = grover_theta(database_size)
    ks = np.asarray(iterations, dtype=float)
    return np.sin((2.0 * ks + 1.0) * theta) ** 2


def classical_success_probs(database_size: int, iterations) -> np.ndarray:
    """Return classical random-search success probabilities for comparison."""
    if database_size <= 0:
        raise ValueError("database_size must be positive")
    ks = np.asarray(iterations, dtype=float)
    return np.clip(ks / float(database_size), 0.0, 1.0)
