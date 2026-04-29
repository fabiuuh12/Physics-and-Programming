# ASF Derivation Notes

## Current status

The ASF equation is currently a phenomenological closure:

```text
laplacian(Phi_eff) = 4 pi G rho + lambda_A laplacian(S_A)
Phi_eff ~= Phi_N + lambda_A S_A
```

This is dimensionally consistent, but not yet derived from first principles.
That is acceptable for the first methods-paper scaffold only if the paper is
honest about the model status.

## Derivation paths to pursue

1. Coarse-grained fluid/MHD stress:
   derive an effective force from unresolved pressure, magnetic, and Reynolds
   stress terms, then ask when it can be represented as `-lambda_A grad(S_A)`.

2. Jeans-equation closure:
   connect unresolved velocity dispersion or anisotropic support to an
   effective acceleration in collisionless halo response.

3. Effective potential fitting:
   define ASF as the lowest-order scalar closure that maps unresolved
   structural diagnostics to potential corrections.

4. Dynamic field extension:
   replace instantaneous `S_A` with
   `partial_t S_A + v dot grad S_A = sources - damping + diffusion`.

## Must-not-overclaim rules

- Do not say ASF modifies fundamental gravity.
- Do not say magnetic fields create mass.
- Do not claim the formula is true before controlled simulations.
- Do not fit rotation curves without a mass-only control and parameter penalty.
