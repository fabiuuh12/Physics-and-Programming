# Physics and Programming

One collection of interactive physics demos, camera experiments, mathematics tools, and security/file utilities. Everything lives inside [`CompPhysics/`](CompPhysics/README.md), organized by purpose.

Open the numbered program menu from this folder:

```bash
python3 explore.py
```

Use `python` on Windows. The menu needs only Python 3.10+; individual programs may need C++ tools, Python packages, or a webcam.

| Start here | What it covers |
| --- | --- |
| [Getting started](CompPhysics/docs/GETTING_STARTED.md) | Install dependencies and run programs |
| [Folder map](CompPhysics/docs/STRUCTURE.md) | Where source, resources, configuration, and outputs belong |
| [Program catalog](CompPhysics/docs/CATALOG.md) | Links to every demo, utility, and supporting module |
| [Physics subjects](CompPhysics/src/simulations/README.md) | Browse simulations by subject |
| [CyberTools](CompPhysics/src/cybertools/README.md) | Browse security and file utilities within the same collection |

```bash
python3 explore.py list gravity
python3 explore.py list cybertools
python3 explore.py info three_body_problem_viz
python3 explore.py run three_body_problem_viz
python3 explore.py doctor
```

Run checks with `python3 CompPhysics/scripts/check.py` after installing `CompPhysics/config/requirements-dev.txt`. Builds, caches, local recordings, and reports belong under `CompPhysics/output/`.
