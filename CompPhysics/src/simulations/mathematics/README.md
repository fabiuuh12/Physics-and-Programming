# Mathematics

Interactive math-oriented tools and supporting documents.

## Contents

- `CompPhysics/docs/reference/mathematics/`: math and physics reference documents
  - [Physics Calc (PDF)](../../../docs/reference/mathematics/Physics%20Calc.pdf) ([LaTeX source](../../../docs/reference/mathematics/Physics%20Calc.tex)): a standalone LaTeX book covering mathematical foundations, classical and modern physics, derivations, worked examples, and review problems with solutions.
- `equation_solver_animation.py`: black-screen algebra animation for equations, derivatives, and integrals

## Run

```bash
python3 CompPhysics/src/simulations/mathematics/equation_solver_animation.py
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

## Build Physics Calc

Create `CompPhysics/output/reports/` first. With a standard LaTeX installation, run from `CompPhysics/docs/reference/mathematics`:

```bash
pdflatex -output-directory="../../../output/reports" -interaction=nonstopmode -halt-on-error "Physics Calc.tex"
pdflatex -output-directory="../../../output/reports" -interaction=nonstopmode -halt-on-error "Physics Calc.tex"
```

The second pass populates the table of contents. The source is self-contained and can also be uploaded to an online LaTeX editor.
