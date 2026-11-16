#!/usr/bin/env python3
"""Launch the C++ power plant and its separate Python webcam hand tracker."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parents[3]


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def build_plant():
    target = 'nuclear_power_plant_viz_cpp'
    if sys.platform == 'win32':
        build = PROJECT_DIR / 'output' / 'build'
        subprocess.run(['cmake', '-S', str(PROJECT_DIR), '-B', str(build)], check=True)
        subprocess.run(['cmake', '--build', str(build), '--config', 'Release', '--target', target], check=True)
        candidates = (build / 'Release' / (target + '.exe'), build / (target + '.exe'))
        return next(path for path in candidates if path.is_file())
    subprocess.run(['bash', str(PROJECT_DIR / 'scripts' / 'run'), '--build-only', target], check=True)
    return PROJECT_DIR / 'output' / 'native' / target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mouse-only', action='store_true', help='Run without opening the webcam')
    args = parser.parse_args()
    use_camera = not args.mouse_only
    if use_camera:
        missing = [name for name in ('cv2', 'mediapipe', 'numpy') if importlib.util.find_spec(name) is None]
        if missing:
            print('Hand tracking needs: ' + ', '.join(missing) + '. Install CompPhysics/config/requirements.txt.\nStarting with mouse controls.', flush=True)
            use_camera = False
    executable = build_plant()
    simulation = tracker = None
    try:
        simulation = subprocess.Popen([str(executable)], cwd=PROJECT_DIR)
        if use_camera:
            tracker = subprocess.Popen([sys.executable, str(PROJECT_DIR / 'src' / 'vision' / 'hand_biomechanics_bridge.py')], cwd=PROJECT_DIR)
            print('Move your mirrored hand cursor over the console. Open your fingers, then pinch to grab.\nRelease to let go. Close the plant to stop both processes.', flush=True)
        while simulation.poll() is None:
            if tracker is not None and tracker.poll() is not None:
                print('Webcam tracker stopped. Mouse controls remain available.', flush=True)
                tracker = None
            time.sleep(.1)
        return simulation.returncode
    except KeyboardInterrupt:
        return 130
    finally:
        stop_child(tracker)
        stop_child(simulation)


if __name__ == '__main__':
    raise SystemExit(main())
