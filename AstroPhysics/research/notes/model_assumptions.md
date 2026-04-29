# ASF Model Assumptions

## Core assumptions

- `S_A` is dimensionless.
- `lambda_A` has units of potential, equivalent to velocity squared.
- The correction is only meaningful for unresolved or subgrid structure.
- Uniform `S_A` produces no extra acceleration.
- `lambda_A = 0` recovers the Newtonian-only model.

## V1 field definition

```text
S_A = ln(rho / rho_0)
    + chi_B * B^2 / (2 mu_0 P_gas)
    + chi_w * |curl v|^2 / Omega_0^2
    + chi_T * L_0^2 |grad T|^2 / T_0^2
```

## Numerical assumptions

- The first solver is 2D and controlled, not a full galaxy simulation.
- Boundary values are fixed to zero for the simple Poisson solve.
- Fields are clipped with small positive floors to avoid division by zero.
- Smoothing can be used for numerical stability but must be reported.

## Failure modes

- The correction requires `epsilon_A` near or above unity with no physical
  justification.
- Results depend more on smoothing or boundary conditions than on structure.
- The model duplicates effects that the hydrodynamic/MHD solver already
  resolves.
- Particle energy drift is dominated by numerical error.
