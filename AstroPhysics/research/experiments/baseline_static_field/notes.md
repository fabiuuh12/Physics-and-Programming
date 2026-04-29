# Baseline Static Field Experiment

## Goal

Hold the mass density fixed and compare Newtonian-only fields against ASF
effective fields.

## Expected outputs

- `fields.npz`: numerical arrays for density, structure, potentials, and
  accelerations.
- `summary.json`: residuals and scalar diagnostics.
- PNG figures showing `rho`, `S_A`, `Phi_N`, `Phi_eff`, and acceleration change.

## Failure criteria

- Non-finite values.
- Poisson residual does not decrease.
- Uniform or nearly uniform `S_A` produces significant extra acceleration.
