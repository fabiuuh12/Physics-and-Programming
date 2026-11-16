#!/usr/bin/env python3
"""Replay synthetic webcam packets through the rendered C++ plant, without a camera.

Build with scripts/run --build-only nuclear_power_plant_viz first. Requires a
working desktop. Coordinates target the default 1400x880 camera and console.
"""
from pathlib import Path
import os
import re
import socket
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[1]


def replay(shutdown):
    executable = PROJECT / 'output' / 'native' / 'nuclear_power_plant_viz_cpp'
    report = PROJECT / 'output' / 'reports'
    report.mkdir(parents=True, exist_ok=True)
    log = report / ('plant_shutdown_input.log' if shutdown else 'plant_dual_hand_input.log')
    process = None
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender, log.open('w') as output:
        try:
            process = subprocess.Popen([str(executable)], stdout=output, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, COMPPHYSICS_SMOKE_FRAMES='180'))
            start = time.monotonic()
            while process.poll() is None and time.monotonic() - start < 12:
                elapsed = time.monotonic() - start
                pull = min(1, max(0, (elapsed - 1) / .5))
                down = int(.7 < elapsed < 2) if not shutdown else int(elapsed > .7)

                def hand(x, y):
                    landmarks = [.5, .5, 0] * 21
                    for index in (4, 8):
                        landmarks[index * 3:index * 3 + 3] = [x / 1400, y / 880, 0]
                    return [1, down, 1] + landmarks

                left = hand(596, 577) if shutdown else hand(449, 506 + 45 * pull)
                right = [0, 0, 0] + [.5, .5, 0] * 21 if shutdown else hand(531, 547 - 22 * pull)
                payload = ','.join(map(str, [elapsed] + left + right)).encode()
                sender.sendto(payload, ('127.0.0.1', 50515))
                time.sleep(.016)
            code = process.wait(timeout=3)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    text = log.read_text()
    match = re.search(r'PLANT: rods=([\d.]+) load=([\d.]+) shutdown=(\d)', text)
    assert code == 0 and match, f'Plant failed; see {log}'
    assert float(match[1]) > .65, 'Rod lever / shutdown failed'
    assert match[3] == str(int(shutdown)), 'Unexpected shutdown state'
    if not shutdown:
        assert float(match[2]) < .8, 'Load dial did not turn'
    print('PASS:', 'pinch shutdown' if shutdown else 'simultaneous two-hand lever and dial', match.group(0))


if __name__ == '__main__':
    replay(False)
    replay(True)
