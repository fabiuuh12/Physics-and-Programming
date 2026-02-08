# Physics-and-Programming

A collection of small Python projects spanning physics visualizations and cybersecurity utilities.

## Folders

- `QuantumStuff/`
  - Animated physics visualizations (quantum, relativity, thermodynamics, classical mechanics, EM, etc.).
  - Includes the LaTeX compendium `QuantumStuff_Equations.tex` and compiled PDF `QuantumStuff_Equations.pdf`.

- `CyberTools/`
  - Small, safe cybersecurity utilities for analysis and hygiene (no offensive tooling).
  - Examples include hash checks, OSINT helper, URL heuristics, password tools, log parsing, and file audits.

- `external/`
  - External or third-party items (kept separate from core project code).

## Quick Start

### Run a visualization
```powershell
cd QuantumStuff
python quantum_tunneling_viz.py
```

### Compile the equations PDF
```powershell
cd QuantumStuff
pdflatex QuantumStuff_Equations.tex
```

### Run a cybersecurity tool
```powershell
cd CyberTools
python "Password Strength Checker.py" --password "example"
```

## Notes
- Most scripts are self-contained and use `numpy` + `matplotlib`.
- CyberTools scripts are designed for local, defensive use and simple analysis.
