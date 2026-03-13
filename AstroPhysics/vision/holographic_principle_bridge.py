#!/usr/bin/env python3
"""
Bridge launcher for the holographic principle visualization.

This intentionally reuses the existing webcam finger tracker and its
`AstroPhysics/vision/live_controls.txt` export as the bridge contract
consumed by the C++ raylib simulation.
"""

from __future__ import annotations

from webcam_finger_tracker import main


if __name__ == "__main__":
    raise SystemExit(main())
