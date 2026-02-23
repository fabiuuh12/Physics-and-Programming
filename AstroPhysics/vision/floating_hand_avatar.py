"""
Compatibility launcher.

The hand-tracking bridge moved to:
AstroPhysics/DefensiveSys/hand_turret_sim.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    astro_root = Path(__file__).resolve().parents[1]
    if str(astro_root) not in sys.path:
        sys.path.insert(0, str(astro_root))
    from DefensiveSys.hand_turret_sim import main as new_main

    return int(new_main())


if __name__ == "__main__":
    raise SystemExit(main())
