"""Opt-in desktop checks for the spacetime lab. Requires Pillow.

Build: ./CompPhysics/scripts/run --build-only spacetime_geometry_viz
Run: python3 CompPhysics/tests/spacetime_geometry_graphics.py
"""
import json
from pathlib import Path
import subprocess


def main():
    from PIL import Image, ImageChops, ImageStat
    root = Path(__file__).resolve().parents[1]
    executable = root / 'output/native/spacetime_geometry_viz_cpp'
    output = root / 'output/reports/spacetime_geometry'
    output.mkdir(parents=True, exist_ok=True)
    profiles = [(f'{scene}_lattice_{g}', ['--scene', scene, '--lattice', str(g)])
                for scene in ('empty', 'mixed', 'planet', 'star', 'blackhole', 'wormhole')
                for g in range(2)]
    scenes, results = {}, []
    for name, options in profiles:
        path = output / f'{name}.png'
        path.unlink(missing_ok=True)
        run = subprocess.run([str(executable), *options, '--capture', str(path)],
                             capture_output=True, text=True, timeout=30)
        log = run.stdout + run.stderr
        (output / f'{name}.log').write_text(log)
        assert run.returncode == 0 and path.exists(), log
        assert 'ERROR:' not in log, log
        with Image.open(path) as image:
            assert image.size == (1440, 900)
            scene = image.convert('RGB').crop((300, 240, 1050, 730))
        assert max(ImageStat.Stat(scene).stddev) > 5, f'Blank scene: {name}'
        scenes[name] = scene
        results.append({'profile': name, 'passed': True})
        print(f'PASS {name}', flush=True)
    for scene in ('empty', 'mixed', 'planet', 'star', 'blackhole', 'wormhole'):
        difference = ImageChops.difference(scenes[f'{scene}_lattice_0'], scenes[f'{scene}_lattice_1'])
        assert max(ImageStat.Stat(difference).mean) > 1, f'Lattice toggle has no visible effect: {scene}'
    difference = ImageChops.difference(scenes['empty_lattice_0'], scenes['mixed_lattice_0'])
    assert max(ImageStat.Stat(difference).mean) > 2, 'Placing objects must visibly deform the grid'
    invalid = subprocess.run([str(executable), '--scene', 'invalid'], capture_output=True, text=True)
    assert invalid.returncode == 2, 'Invalid scene must be rejected before opening a window'
    (output / 'results.json').write_text(json.dumps(results, indent=2))
    print(f'{len(results)} graphics profiles passed; {output}')


if __name__ == '__main__':
    main()
