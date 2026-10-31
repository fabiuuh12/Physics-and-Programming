"""Opt-in desktop validation for all 12 Cosmic Atlas tracks and 60 stages.

Build with scripts/run --build-only cosmic_evolution_viz, then execute this file.
Requires Pillow (included in config/requirements.txt). Outputs remain in output/.
"""
import json
from pathlib import Path
import subprocess


def main():
    from PIL import Image, ImageChops, ImageStat
    root = Path(__file__).resolve().parents[1]
    executable = root / 'output/native/cosmic_evolution_viz_cpp'
    output = root / 'output/reports/cosmic_atlas'
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for chapter in range(4):
        for variant in range(3):
            scenes = []
            for stage in range(5):
                path = output / f'{chapter}_{variant}_{stage}.png'
                path.unlink(missing_ok=True)
                run = subprocess.run([
                    str(executable), '--chapter', str(chapter), '--variant', str(variant),
                    '--progress', str(stage * .2 + .1), '--capture', str(path),
                ], capture_output=True, text=True, timeout=30)
                log = run.stdout + run.stderr
                (output / f'{path.stem}.log').write_text(log)
                assert run.returncode == 0 and path.exists(), log
                assert 'Fragment shader compiled successfully' in log and 'ERROR:' not in log, log
                with Image.open(path) as image:
                    assert image.size == (1440, 900)
                    scene = image.convert('RGB').crop((100, 270, 940, 610))
                assert max(ImageStat.Stat(scene).stddev) > 3, f'Blank scene: {path}'
                scenes.append(scene)
                results.append({'chapter': chapter, 'variant': variant, 'stage': stage, 'passed': True})
            difference = ImageStat.Stat(ImageChops.difference(scenes[0], scenes[-1]))
            assert max(difference.mean) > 1, f'Track has no visual evolution: {chapter}/{variant}'
            print(f'PASS chapter={chapter} variant={variant}: 5 rendered stages', flush=True)
    (output / 'results.json').write_text(json.dumps(results, indent=2))
    print(f'{len(results)} rendered stages passed; {output}')


if __name__ == '__main__':
    main()
