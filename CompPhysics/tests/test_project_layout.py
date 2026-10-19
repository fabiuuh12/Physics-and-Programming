"""Check shared resource resolution without loading camera or GUI dependencies."""
import ast
from pathlib import Path
import warnings

PROJECT = Path(__file__).resolve().parents[1]


def test_camera_resources_and_runtime_paths():
    names = {'MODULE_DIR', 'SCRIPT_DIR', 'PROJECT_DIR', 'MODEL_DIR', 'MODEL_CANDIDATES',
             'MODEL_PATH', 'DEFAULT_POSE_MODEL', 'DEFAULT_DATABASE', 'CONTROL_OUTPUT_PATH'}
    checked = 0
    for script in (PROJECT / 'src').rglob('*.py'):
        namespace = {'Path': Path, '__file__': str(script)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            tree = ast.parse(script.read_text(encoding='utf-8-sig'))
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
            elif isinstance(node, ast.AnnAssign):
                target = node.target
            else:
                continue
            if isinstance(target, ast.Name) and target.id in names:
                namespace[target.id] = eval(compile(ast.Expression(node.value), str(script), 'eval'), namespace)
        for name in ('MODEL_PATH', 'DEFAULT_POSE_MODEL'):
            if name in namespace:
                model = namespace[name]
                assert model.is_file(), (script, model)
                assert model.parent == PROJECT / 'resources/models'
                checked += 1
        for name in ('DEFAULT_DATABASE', 'CONTROL_OUTPUT_PATH'):
            if name in namespace:
                assert namespace[name].is_relative_to(PROJECT / 'output'), script
    assert checked >= 10


def test_project_root_and_sources_are_free_of_generated_clutter():
    assert {p.name for p in PROJECT.iterdir() if p.is_dir()} == {
        'src', 'resources', 'vendor', 'config', 'scripts', 'docs', 'tests', 'output'
    }
    assert not list((PROJECT / 'src').rglob('__pycache__'))
    assert (PROJECT / 'resources/fonts/Inter-Regular.ttf').is_file()
