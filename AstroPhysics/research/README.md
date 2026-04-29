# Astrophysical Structure Field Research Workspace

This workspace develops the Astrophysical Structure Field (ASF) model as a
methods-paper project: a phenomenological subgrid/effective-potential
framework for testing whether unresolved baryonic structure can change
dwarf-galaxy halo diagnostics.

The ASF model is not treated here as a new law of gravity. It is a controlled
computational hypothesis:

```text
grad^2 Phi_eff = 4 pi G rho + lambda_A grad^2 S_A
Phi_eff ~= Phi_N + lambda_A S_A
a_eff = -grad Phi_eff
```

## Layout

- `paper/`: LaTeX manuscript, bibliography, equation macros, and figure folder.
- `notes/`: derivation notes, assumptions, literature map, and presentation notes.
- `python/`: NumPy/Matplotlib prototype and runnable scripts.
- `cpp/`: C++17 standard-library prototype and numerical checks.
- `experiments/`: JSON configs and run notes.
- `results/`: generated data and plots.
- `tests/`: Python test suite.

## Python quick start

Run from repository root:

```bash
PYTHONPATH=AstroPhysics/research/python python3 AstroPhysics/research/python/scripts/run_static_field.py \
  --config AstroPhysics/research/experiments/baseline_static_field/config.json

PYTHONPATH=AstroPhysics/research/python python3 AstroPhysics/research/python/scripts/run_particles.py \
  --config AstroPhysics/research/experiments/particle_orbits/config.json

PYTHONPATH=AstroPhysics/research/python python3 -m pytest AstroPhysics/research/tests -q
```

The current environment has NumPy and Matplotlib available. SciPy is not
required for v1.

## C++ quick start

The C++ prototype avoids CMake for v1 and can compile directly with the system
C++17 compiler:

```bash
c++ -std=c++17 -O2 -I AstroPhysics/research/cpp/include \
  AstroPhysics/research/cpp/src/asf_sim.cpp \
  AstroPhysics/research/cpp/src/run_static_field.cpp \
  -o /tmp/asf_static

/tmp/asf_static AstroPhysics/research/experiments/baseline_static_field/config.json \
  AstroPhysics/research/results/cpp_static.csv

c++ -std=c++17 -O2 -I AstroPhysics/research/cpp/include \
  AstroPhysics/research/cpp/src/asf_sim.cpp \
  AstroPhysics/research/cpp/tests/asf_numeric_tests.cpp \
  -o /tmp/asf_tests

/tmp/asf_tests
```

## Paper workflow

The manuscript source is `paper/main.tex`. This machine currently does not have
`pdflatex` or `latexmk` on PATH, so compilation is deferred until a TeX
distribution is installed. The source is still kept ready for standard LaTeX
toolchains.

## Research discipline

Every claim should be backed by one of:

- a derivation or controlled approximation,
- a numerical experiment with a mass-only control,
- a parameter scan showing a constrained viable range,
- a failure result that narrows the model.

The default failure criteria are numerical instability, unphysical parameter
values, worsened rotation-curve diagnostics, conservation problems, or
double-counting resolved hydrodynamic/MHD physics.
