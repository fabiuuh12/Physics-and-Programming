# Mathematics

Interactive math-oriented tools and supporting documents.

## Contents

- `docs/`: moved math/physics document folder
- `equation_solver_animation.py`: black-screen algebra animation for equations, derivatives, and integrals

## Run

```bash
python3 AstroPhysics/mathematics/equation_solver_animation.py
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
