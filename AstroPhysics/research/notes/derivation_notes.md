# Derivation Notes

## Dynamics

The planar CR3BP equations are used because they are standard, cislunar
relevant, and feasible on a laptop. They model a massless spacecraft under the
gravity of Earth and Moon in a rotating nondimensional frame.

## EKF

The EKF propagates a nonlinear state model and uses a local linearization for
covariance prediction and measurement update. V1 uses finite-difference
Jacobians for the dynamics and analytic Jacobians for range/bearing.

## Risk score

The risk score combines:

- covariance size,
- measurement geometry,
- missed measurements,
- lighting/visibility loss.

The score is dimensionless after scaling by `r0`, `v0`, and dimensionless
weights.

## Adaptation rule

The risk score adapts either process noise or measurement noise. The adaptation
is clipped to prevent runaway covariance inflation.
