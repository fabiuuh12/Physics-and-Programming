# Configuration

Install these from the repository root, preferably inside a virtual environment:

```bash
python3 -m pip install -r CompPhysics/config/requirements.txt
python3 -m pip install -r CompPhysics/config/requirements-dev.txt
```

`requirements.txt` covers shared runtime packages. `requirements-dev.txt` covers numerical tests. Individual programs may have additional platform requirements. C++ targets remain in `../CMakeLists.txt`; pytest settings are in the repository-root `pytest.ini`. See [Getting started](../docs/GETTING_STARTED.md).

`editor/vscode/settings.json` is a portable VS Code settings template. Copy its entries into the repository-root `.vscode/settings.json` to keep editor builds in `output/build/`. The active local settings have been updated too.
