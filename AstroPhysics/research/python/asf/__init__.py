"""Astrophysical Structure Field numerical prototype."""

from .config import load_config
from .fields import FieldState, build_controlled_fields, compute_structure_field
from .numerics import gradient, laplacian, poisson_jacobi, poisson_residual
from .particles import ParticleState, initialize_ring_particles, integrate_particles

__all__ = [
    "FieldState",
    "ParticleState",
    "build_controlled_fields",
    "compute_structure_field",
    "gradient",
    "initialize_ring_particles",
    "integrate_particles",
    "laplacian",
    "load_config",
    "poisson_jacobi",
    "poisson_residual",
]
