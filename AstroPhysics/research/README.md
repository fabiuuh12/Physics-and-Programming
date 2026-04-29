# Risk-Aware Cislunar Navigation Research Workspace

This workspace develops a laptop-scale computational aerospace research project:

**Risk-Aware Adaptive Extended Kalman Filtering for Autonomous Cislunar Spacecraft Navigation**

The project simulates a spacecraft in the planar circular restricted three-body
problem (CR3BP), estimates its state with EKF variants, and tests a custom
navigation risk score:

```text
R_nav = w_r sigma_r/r0 + w_v sigma_v/v0 + w_g/(Gamma + epsilon) + w_m M + w_l L
```

The goal is not new physics. The goal is a disciplined autonomous navigation
study: does a risk-aware adaptive EKF behave better than a fixed-noise EKF under
missed measurements, poor geometry, and lighting/visibility loss?

## Layout

- `paper/`: LaTeX research paper, equation explainer, bibliography, figures.
- `notes/`: derivation notes, assumptions, literature map, professor pitch.
- `python/`: CR3BP dynamics, measurement model, EKF, risk score, experiments.
- `cpp/`: C++17 mirror of the core dynamics and risk equation.
- `experiments/`: JSON scenarios for clean and stressed navigation cases.
- `results/`: generated plots/data.
- `tests/`: Python numerical tests.

## Python quick start

```bash
PYTHONPATH=AstroPhysics/research/python python3 AstroPhysics/research/python/scripts/run_experiment.py \
  --config AstroPhysics/research/experiments/baseline/config.json

PYTHONPATH=AstroPhysics/research/python python3 -m pytest AstroPhysics/research/tests -q
```

Run the parameter scan:

```bash
PYTHONPATH=AstroPhysics/research/python python3 AstroPhysics/research/python/scripts/run_parameter_scan.py \
  --config AstroPhysics/research/experiments/parameter_scan/config.json
```

Analyze the parameter scan:

```bash
PYTHONPATH=AstroPhysics/research/python python3 AstroPhysics/research/python/scripts/analyze_parameter_scan.py \
  --summary AstroPhysics/research/results/parameter_scan/parameter_scan_summary.csv \
  --out AstroPhysics/research/results/parameter_scan/analysis
```

## C++ quick start

```bash
c++ -std=c++17 -O2 -I AstroPhysics/research/cpp/include \
  AstroPhysics/research/cpp/src/cislunar_nav.cpp \
  AstroPhysics/research/cpp/tests/cislunar_numeric_tests.cpp \
  -o /private/tmp/cislunar_cpp_tests

/private/tmp/cislunar_cpp_tests
```

## Paper compile

```bash
cd AstroPhysics/research/paper
TEXMFVAR=/private/tmp/asf_texmf_var PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH" pdflatex main.tex
TEXMFVAR=/private/tmp/asf_texmf_var PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH" bibtex main
TEXMFVAR=/private/tmp/asf_texmf_var PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH" pdflatex main.tex
TEXMFVAR=/private/tmp/asf_texmf_var PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH" pdflatex main.tex
```

Replace `main` with `equation_explainer` or `formula_deep_dive` to compile the
supporting explanation papers.
