"""Compile and run the rendering-independent C++ model regression checks."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_cpp_numerical_models(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++17 compiler is required for the numerical model checks")
    source = Path(__file__).with_name("simulation_numerics_tests.cpp")
    executable = tmp_path / "simulation_numerics_tests"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", str(source), "-o", str(executable)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
