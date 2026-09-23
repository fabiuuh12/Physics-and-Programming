# Mathematics

Interactive math-oriented tools and supporting documents.

## Contents

- `docs/`: math and physics reference documents
  - [Physics: A Comprehensive Review (PDF)](docs/Physics_Review_Classical_to_Modern.pdf) ([LaTeX source](docs/Physics_Review_Classical_to_Modern.tex)): a standalone LaTeX book covering mathematical foundations, classical and modern physics, derivations, worked examples, and review problems with solutions.
  - `QuantumStuff_Equations.tex`: a compact companion to the visualization scripts.
- `equation_solver_animation.py`: black-screen algebra animation for equations, derivatives, and integrals

## Run

```bash
python3 CompPhysics/mathematics/equation_solver_animation.py
```

Examples:

- `2*x + 3 = 11`
- `5*x - 7 = 2*x + 8`
- `x/3 + 5 = 9`
- `diff x^3 + 2*x - 5`
- `int x^2 + 3*x`
- `int[0,1] x^2`

Controls:

- `q` or `Esc`: quit
- `Enter`: solve the typed equation
- `p` or `Space`: pause
- `r`: restart animation from the first solving step
- `n`: return to the in-window equation editor
- arrow keys: move the input cursor
- `[` / `]`: slower / faster animation

## Build the physics review

With a standard LaTeX installation, run from `CompPhysics/mathematics/docs`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error Physics_Review_Classical_to_Modern.tex
pdflatex -interaction=nonstopmode -halt-on-error Physics_Review_Classical_to_Modern.tex
```

The second pass populates the table of contents. The source is self-contained and can also be uploaded to an online LaTeX editor.
