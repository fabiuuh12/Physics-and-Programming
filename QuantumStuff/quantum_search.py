import numpy as np


def grover_theta(search_space_size: int) -> float:
    """Return theta where sin(theta) = 1/sqrt(N)."""
    if search_space_size <= 0:
        raise ValueError("search_space_size must be positive")
    return np.arcsin(1 / np.sqrt(search_space_size))


def classical_success_probs(search_space_size: int, k_values: np.ndarray) -> np.ndarray:
    """Classical random-guess success probability ~ k/N (capped at 1)."""
    return np.clip(k_values / search_space_size, 0, 1)


def quantum_success_probs(search_space_size: int, k_values: np.ndarray) -> np.ndarray:
    """Grover-style success probability."""
    theta = grover_theta(search_space_size)
    return np.sin((2 * k_values + 1) * theta) ** 2


def optimal_grover_iterations(search_space_size: int) -> int:
    """k* that maximizes Grover success probability."""
    theta = grover_theta(search_space_size)
    return int(round(np.pi / (4 * theta) - 0.5))
