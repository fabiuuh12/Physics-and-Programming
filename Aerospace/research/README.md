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

Run the full reproducible experiment and analysis pipeline:

```bash
Aerospace/research/run_all_experiments.sh
```

Run one scenario manually:

```bash
PYTHONPATH=Aerospace/research/python python3 Aerospace/research/python/scripts/run_experiment.py \
  --config Aerospace/research/experiments/baseline/config.json

PYTHONPATH=Aerospace/research/python python3 -m pytest Aerospace/research/tests -q
```

Run the parameter scan:

```bash
PYTHONPATH=Aerospace/research/python python3 Aerospace/research/python/scripts/run_parameter_scan.py \
  --config Aerospace/research/experiments/parameter_scan/config.json
```

Analyze the parameter scan:

```bash
PYTHONPATH=Aerospace/research/python python3 Aerospace/research/python/scripts/analyze_parameter_scan.py \
  --summary Aerospace/research/results/parameter_scan/parameter_scan_summary.csv \
  --out Aerospace/research/results/parameter_scan/analysis
```

Analyze controlled scenario outputs:

```bash
PYTHONPATH=Aerospace/research/python python3 Aerospace/research/python/scripts/analyze_scenarios.py \
  --results-root Aerospace/research/results \
  --out Aerospace/research/results/scenario_analysis
```

## C++ quick start

```bash
c++ -std=c++17 -O2 -I Aerospace/research/cpp/include \
  Aerospace/research/cpp/src/cislunar_nav.cpp \
  Aerospace/research/cpp/tests/cislunar_numeric_tests.cpp \
  -o /private/tmp/cislunar_cpp_tests

/private/tmp/cislunar_cpp_tests
```

The C++ folder also includes an interactive raylib visualization:

```bash
c++ -std=c++17 -O2 \
  -I Aerospace/research/cpp/include \
  -I Aerospace/AstroPhysics/third_party/raylib/include \
  Aerospace/research/cpp/src/cislunar_nav.cpp \
  Aerospace/research/cpp/src/cislunar_nav_3d_viz.cpp \
  Aerospace/AstroPhysics/third_party/raylib-src/src/libraylib.a \
  -framework CoreVideo -framework IOKit -framework Cocoa -framework OpenGL \
  -o /private/tmp/cislunar_nav_3d_viz
```

## Paper compile

```bash
cd Aerospace/research/paper
TEXMFVAR=/private/tmp/asf_texmf_var PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH" pdflatex main.tex
TEXMFVAR=/private/tmp/asf_texmf_var PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH" bibtex main
TEXMFVAR=/private/tmp/asf_texmf_var PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH" pdflatex main.tex
TEXMFVAR=/private/tmp/asf_texmf_var PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH" pdflatex main.tex
```

Replace `main` with `equation_explainer` or `formula_deep_dive` to compile the
supporting explanation papers.
