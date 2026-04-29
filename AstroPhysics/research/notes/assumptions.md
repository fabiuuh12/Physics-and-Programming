# Project Assumptions

- V1 uses planar CR3BP in nondimensional Earth-Moon units.
- The state is `[x, y, vx, vy]`.
- Measurements are range and bearing from fixed reference observers.
- Lighting loss is modeled as a scalar visibility penalty, not image rendering.
- The risk equation is a navigation decision metric, not a physical law.
- The adaptive EKF must be compared against a fixed-noise baseline EKF.
- Claims must be based on reproducible numerical experiments.
