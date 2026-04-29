# Professor-Facing Pitch

I am building a computational aerospace research project on autonomous
cislunar navigation. The project simulates a spacecraft in the planar
Earth-Moon CR3BP and compares a standard EKF against risk-aware adaptive EKF
variants.

The new part is a navigation risk score that combines covariance growth,
measurement geometry, missed measurements, and lighting/visibility loss. The
score adapts the filter's process or measurement noise. The research question
is whether this improves robustness under realistic stress cases compared with
a fixed-noise EKF.

This is not a claim of new physics. It is a reproducible navigation simulation
and estimator-comparison study.
