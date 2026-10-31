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
    profiles = [(f'preset_{p}_lattice_{g}', ['--preset', str(p), '--lattice', str(g)])
                for p in range(5) for g in range(2)]
    profiles += [
        ('inside_horizon', ['--preset', '3', '--probe', '.75']),
        ('strong_mass', ['--preset', '3', '--mass', '2.4', '--warp', '2']),
        ('thin_throat', ['--preset', '4', '--mass', '.35', '--impact', '0']),
        ('negative_impact', ['--preset', '3', '--impact', '-3']),
    ]
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
    for p in range(5):
        difference = ImageChops.difference(scenes[f'preset_{p}_lattice_0'], scenes[f'preset_{p}_lattice_1'])
        assert max(ImageStat.Stat(difference).mean) > 1, f'Lattice toggle has no visible effect: {p}'
    invalid = subprocess.run([str(executable), '--mass', 'nan'], capture_output=True, text=True)
    assert invalid.returncode == 2, 'Invalid numeric settings must be rejected before opening a window'
    (output / 'results.json').write_text(json.dumps(results, indent=2))
    print(f'{len(results)} graphics profiles passed; {output}')


if __name__ == '__main__':
    main()
