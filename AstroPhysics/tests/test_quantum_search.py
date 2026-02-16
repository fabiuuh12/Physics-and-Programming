import numpy as np
import pytest

from quantum_search import (
    classical_success_probs,
    grover_theta,
    optimal_grover_iterations,
    quantum_success_probs,
)


def test_quantum_outperforms_classical_at_optimal_iterations():
    N = 1024
    k_star = optimal_grover_iterations(N)
    ks = np.array([k_star])

    classical = classical_success_probs(N, ks)[0]
    quantum = quantum_success_probs(N, ks)[0]

    assert classical == pytest.approx(k_star / N, rel=1e-9)
    assert quantum == pytest.approx(1.0, abs=1e-3)
    assert quantum > classical


def test_theta_matches_definition():
    N = 2048
    theta = grover_theta(N)
    assert np.isclose(np.sin(theta), 1 / np.sqrt(N))
