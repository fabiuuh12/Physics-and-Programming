# Particle Orbit Experiment

## Goal

Compare test-particle trajectories in `Phi_N` and `Phi_eff` using identical
initial conditions.

## Diagnostics

- Mean trajectory separation.
- Maximum trajectory separation.
- Energy drift proxy.
- Orbit plots for Newtonian and ASF runs.

## Failure criteria

- Large energy drift in both controls and ASF runs.
- ASF behavior dominated by grid noise.
- Results vanish unless `lambda_A` is unphysically large.
