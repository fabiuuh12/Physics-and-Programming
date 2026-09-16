"""Opt-in graphics checks for the refreshed C++ demos (requires a desktop).

Build first with ./CompPhysics/run --build-all, then run this script.
Hidden windows execute real draw loops and export PNG previews into /tmp.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--filter", default="", help="Only targets containing this text")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = re.findall(r'add_executable\((\S+) "([^"]+)"\)', (root / "CMakeLists.txt").read_text())
    results = []
    for target, source in rows:
        source_path = root / source
        if args.filter not in target or 'common/studio.h' not in source_path.read_text():
            continue
        executable = root / "build-native" / target
        is_water = source_path.stem == "shallow_water_sandbox_viz"
        preview = Path("/tmp/shallow_water_sandbox.png") if is_water else Path("/tmp") / f"{source_path.stem}_preview.png"
        if preview.exists():
            preview.unlink()
        command = [str(executable)] + (["--smoke-test"] if is_water else [])
        env = dict(os.environ, COMPPHYSICS_SMOKE_FRAMES=str(args.frames))
        try:
            run = subprocess.run(command, cwd=root.parent, env=env, capture_output=True, text=True, timeout=45)
            ok = run.returncode == 0 and preview.exists() and preview.stat().st_size > 1000
            output = run.stdout + run.stderr
        except subprocess.TimeoutExpired:
            ok, output = False, "Graphics process timed out"
        log = Path("/tmp") / f"{source_path.stem}_graphics.log"
        log.write_text(output)
        results.append({"target": target, "passed": ok, "preview": str(preview), "log": str(log)})
        print(f"{'PASS' if ok else 'FAIL'} {target}", flush=True)
    report = Path("/tmp/compphysics-graphics-results.json")
    report.write_text(json.dumps(results, indent=2))
    print(f"{sum(row['passed'] for row in results)}/{len(results)} passed; {report}")
    return 0 if results and all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
