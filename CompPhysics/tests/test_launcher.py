"""Discovery and dispatch checks; no graphics, camera, or network required."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('repository_explore', ROOT / 'explore.py')
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)


def test_catalog_covers_registered_sources_and_distinguishes_modules():
    entries = launcher.catalog()
    assert entries and all(e.path.is_file() for e in entries)
    assert len({e.name for e in entries}) == len(entries)
    assert launcher.select(entries, 'projectile_drag_viz_cpp').target == 'projectile_drag_viz_cpp'
    assert launcher.select(entries, 'CompPhysics/src/vision/two_hand_bridge.py').kind == 'Python script'
    assert launcher.select(entries, 'optics/hand_optics_scenes.py').kind == 'Python module'
    with pytest.raises(ValueError, match='Multiple matches'):
        launcher.select(entries, 'gravity')
    with pytest.raises(ValueError, match='No match'):
        launcher.select(entries, 'does-not-exist')


def test_python_dispatch_preserves_arguments_and_working_directory(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, 'run', lambda cmd, **kw: calls.append((cmd, kw)))
    entry = launcher.select(launcher.catalog(), 'Certificate Expiry Checker.py')
    launcher.run_entry(entry, ['--file', '/tmp/hosts.txt'])
    command, options = calls[0]
    assert command == [sys.executable, str(entry.path), '--file', '/tmp/hosts.txt']
    assert options['cwd'] == ROOT and options['check']
    assert options['env']['COMPPHYSICS_ROOT'] == str(ROOT / 'CompPhysics')
    assert options['env']['PYTHONPYCACHEPREFIX'] == str(ROOT / 'CompPhysics/output/cache/python')


def test_cpp_dispatch_builds_selected_target_and_dry_run_is_inert(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    entry = launcher.select(launcher.catalog(), 'projectile_drag_viz')
    launcher.run_entry(entry, [], dry_run=True)
    assert not calls
    launcher.run_entry(entry, [])
    assert calls[0][:2] == ['cmake', '-S']
    assert calls[1][-2:] == ['--target', entry.target]
    assert Path(calls[2][0]).stem == entry.target
    module = launcher.select(launcher.catalog(), 'optics/hand_optics_scenes.py')
    with pytest.raises(ValueError, match='supporting module'):
        launcher.run_entry(module, [])


def test_reorganized_sources_and_legacy_launcher_names():
    entries = launcher.catalog()
    demo = launcher.select(entries, 'gravity/three_body_problem_viz.cpp')
    assert demo.path == ROOT / 'CompPhysics/src/simulations/gravity/three_body_problem_viz.cpp'
    assert demo.category == 'simulations/gravity'
    assert launcher.select(entries, 'simulations/gravity/three_body_problem_viz.cpp') == demo
    defense = launcher.select(entries, 'DefensiveSys/hand_turret_sim.py')
    assert defense.path.parent.name == 'defensive_systems'
    tool = launcher.select(entries, 'CompPhysics/CyberTools/network/Certificate Expiry Checker.py')
    assert tool.path == ROOT / 'CompPhysics/src/cybertools/network/Certificate Expiry Checker.py'
    assert tool.category == 'cybertools/network'
    assert launcher.select(entries, 'quantum_search.py').path.parent == ROOT / 'CompPhysics/src/simulations/quantum'
    # Every authored standalone C++ source remains registered after the moves.
    expected = set((ROOT / 'CompPhysics/src/simulations').rglob('*.cpp'))
    expected.update((ROOT / 'CompPhysics/src/vision').glob('*.cpp'))
    assert {entry.path for entry in entries if entry.target} == expected
