# Simulation Design

## V1 controlled experiment

The first publishable test is not a full galaxy simulation. It is a controlled
method experiment:

1. Construct the same mass density field `rho` in two runs.
2. Construct a smooth `S_A` in the control run.
3. Construct a structured `S_A` in the ASF run using magnetic support,
   vorticity, and thermal-gradient terms.
4. Solve for `Phi_N`.
5. Compute `Phi_eff = Phi_N + lambda_A S_A`.
6. Compare acceleration fields, orbits, and rotation diagnostics.

## Required controls

- `lambda_A = 0` must exactly recover `Phi_N`.
- Uniform `S_A` must produce zero extra acceleration to finite-difference
  precision.
- The same initial particles must be used for all orbit comparisons.
- Parameter scans must report failed regions, not only successful examples.

## Core outputs

- Potential maps.
- Acceleration-difference maps.
- Radial circular-velocity curves.
- Orbit divergence plots.
- Poisson residual history.
- Parameter scan table.

## PhD-level upgrade path

1. Replace hand-designed fields with snapshots from controlled hydro/MHD runs.
2. Calibrate `S_A` against unresolved stress or pressure support.
3. Compare with synthetic observations and rotation-curve inference.
4. Add dynamic `S_A` evolution and conservation diagnostics.
