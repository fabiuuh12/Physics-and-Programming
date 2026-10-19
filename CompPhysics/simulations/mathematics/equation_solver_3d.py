from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import numpy as np


@dataclass
class Model3DSpec:
    kind: str
    title: str
    expression_text: str
    points: list[tuple[float, float, float, float]]
    edges: list[tuple[int, int]]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    estimate: float | None = None


@dataclass
class Model3DView:
    yaw: float = math.radians(34.0)
    pitch: float = math.radians(58.0)
    zoom: float = 1.0
    slice_axis: str | None = None
    slice_value: float = 0.5


SUBSCRIPT_TRANSLATION = str.maketrans("₋₀₁₂₃₄₅₆₇₈₉", "-0123456789")
SUPERSCRIPT_TRANSLATION = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")


def parse_bound_piece(
    piece: str,
    parse_constant_fraction: Callable[[str], Fraction],
    normalize_input_text: Callable[[str], str],
) -> tuple[str, float, float]:
    match = re.match(r"^\s*([xyz])\s*=\s*([^:]+)\s*:\s*([^:]+)\s*$", piece, re.IGNORECASE)
    if not match:
        raise ValueError("Bounds must look like x=0:1.")
    name = match.group(1).lower()
    low = float(parse_constant_fraction(normalize_input_text(match.group(2).strip())))
    high = float(parse_constant_fraction(normalize_input_text(match.group(3).strip())))
    if abs(high - low) < 1e-9:
        raise ValueError("3D bounds need nonzero width.")
    return name, min(low, high), max(low, high)


def parse_surface_request(
    text: str,
    split_top_level_commas: Callable[[str], list[str]],
    parse_constant_fraction: Callable[[str], Fraction],
    normalize_input_text: Callable[[str], str],
) -> tuple[dict[str, tuple[float, float]], str]:
    match = re.match(r"^(?:surface|surface3d|plot3d|graph3d)(?:\[(.+?)\])?\s+(.+)$", text.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("Surface input must look like surface[x=-3:3,y=-3:3] z=sin(x*y).")
    bounds = {"x": (-4.0, 4.0), "y": (-4.0, 4.0)}
    if match.group(1):
        for piece in split_top_level_commas(match.group(1)):
            name, low, high = parse_bound_piece(piece, parse_constant_fraction, normalize_input_text)
            if name not in ("x", "y"):
                raise ValueError("Surface bounds only support x and y.")
            bounds[name] = (low, high)
    expression = match.group(2).strip()
    expression = re.sub(r"^z\s*=", "", expression, count=1, flags=re.IGNORECASE).strip()
    return bounds, expression


def parse_triple_request(
    text: str,
    split_top_level_commas: Callable[[str], list[str]],
    parse_constant_fraction: Callable[[str], Fraction],
    normalize_input_text: Callable[[str], str],
) -> tuple[dict[str, tuple[float, float]], str]:
    match = re.match(r"^triple\[(.+?)\]\s+(.+)$", text.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("Triple integral input must look like triple[x=0:1,y=0:1,z=0:1] x*y*z.")
    bounds: dict[str, tuple[float, float]] = {}
    for piece in split_top_level_commas(match.group(1)):
        name, low, high = parse_bound_piece(piece, parse_constant_fraction, normalize_input_text)
        bounds[name] = (low, high)
    missing = [name for name in ("x", "y", "z") if name not in bounds]
    if missing:
        raise ValueError("Triple integrals need x, y, and z bounds.")
    return bounds, match.group(2).strip()


def parse_unicode_bound(raw: str, translation: dict[int, str], normalize_input_text: Callable[[str], str]) -> str:
    text = raw.translate(translation).strip()
    return normalize_input_text(text)


def parse_iterated_triple_request(
    text: str,
    parse_constant_fraction: Callable[[str], Fraction],
    normalize_input_text: Callable[[str], str],
) -> tuple[dict[str, tuple[float, float]], str]:
    raw = text.strip()
    if "∫" not in raw:
        raise ValueError("Iterated triple integral must use integral signs.")
    integral_matches = list(re.finditer(r"∫\s*([₋₀₁₂₃₄₅₆₇₈₉πΠeE.+-]*)\s*([⁻⁰¹²³⁴⁵⁶⁷⁸⁹πΠeE.+-]*)", raw))
    if len(integral_matches) < 3:
        raise ValueError("Use three integral signs for a triple integral.")
    bounds_list: list[tuple[float, float]] = []
    for match in integral_matches[:3]:
        low_text = parse_unicode_bound(match.group(1), SUBSCRIPT_TRANSLATION, normalize_input_text)
        high_text = parse_unicode_bound(match.group(2), SUPERSCRIPT_TRANSLATION, normalize_input_text)
        if not low_text or not high_text:
            raise ValueError("Each integral sign needs lower and upper bounds.")
        low = float(parse_constant_fraction(low_text))
        high = float(parse_constant_fraction(high_text))
        bounds_list.append((min(low, high), max(low, high)))

    body = raw[integral_matches[2].end():].strip()
    diff_matches = re.findall(r"d\s*([xyz])", body, re.IGNORECASE)
    if len(diff_matches) < 3:
        raise ValueError("End iterated integrals with differentials like dz dy dx.")
    expression = re.sub(r"(?:\s*d\s*[xyz]){3}\s*$", "", body, flags=re.IGNORECASE).strip()
    variables_outer_to_inner = [var.lower() for var in reversed(diff_matches[-3:])]
    bounds = {var: bound for var, bound in zip(variables_outer_to_inner, bounds_list)}
    missing = [name for name in ("x", "y", "z") if name not in bounds]
    if missing:
        raise ValueError("Triple integral differentials must include dx, dy, and dz.")
    return bounds, expression


def build_surface_model(
    bounds: dict[str, tuple[float, float]],
    expression_text: str,
    parse_numeric_expression: Callable[[str], ast.AST],
    evaluate_numeric_expression_vars: Callable[[ast.AST, dict[str, float]], float],
) -> Model3DSpec:
    node = parse_numeric_expression(expression_text)
    samples = 28
    x_values = np.linspace(bounds["x"][0], bounds["x"][1], samples)
    y_values = np.linspace(bounds["y"][0], bounds["y"][1], samples)
    raw: list[tuple[float, float, float, float]] = []
    finite_z: list[float] = []
    for x in x_values:
        for y in y_values:
            try:
                z = evaluate_numeric_expression_vars(node, {"x": float(x), "y": float(y)})
            except Exception:
                z = float("nan")
            if math.isfinite(z) and abs(z) < 1e5:
                raw.append((float(x), float(y), float(z), float(z)))
                finite_z.append(float(z))
            else:
                raw.append((float(x), float(y), float("nan"), 0.0))
    if not finite_z:
        raise ValueError("Could not sample this surface.")
    z_low = float(np.percentile(finite_z, 5))
    z_high = float(np.percentile(finite_z, 95))
    if abs(z_high - z_low) < 1e-9:
        z_low -= 1.0
        z_high += 1.0
    points = [(x, y, min(max(z, z_low), z_high), value) for x, y, z, value in raw if math.isfinite(z)]
    index_by_xy = {(round(x, 6), round(y, 6)): idx for idx, (x, y, _, _) in enumerate(points)}
    edges: list[tuple[int, int]] = []
    for ix in range(samples):
        for iy in range(samples):
            key = (round(float(x_values[ix]), 6), round(float(y_values[iy]), 6))
            if key not in index_by_xy:
                continue
            here = index_by_xy[key]
            if ix + 1 < samples:
                next_key = (round(float(x_values[ix + 1]), 6), round(float(y_values[iy]), 6))
                if next_key in index_by_xy:
                    edges.append((here, index_by_xy[next_key]))
            if iy + 1 < samples:
                next_key = (round(float(x_values[ix]), 6), round(float(y_values[iy + 1]), 6))
                if next_key in index_by_xy:
                    edges.append((here, index_by_xy[next_key]))
    return Model3DSpec("surface", "3D Surface", expression_text, points, edges, bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1], z_low, z_high)


def build_triple_model(
    bounds: dict[str, tuple[float, float]],
    expression_text: str,
    parse_numeric_expression: Callable[[str], ast.AST],
    evaluate_numeric_expression_vars: Callable[[ast.AST, dict[str, float]], float],
) -> Model3DSpec:
    node = parse_numeric_expression(expression_text)
    samples = 8
    dx = (bounds["x"][1] - bounds["x"][0]) / samples
    dy = (bounds["y"][1] - bounds["y"][0]) / samples
    dz = (bounds["z"][1] - bounds["z"][0]) / samples
    xs = np.linspace(bounds["x"][0] + dx * 0.5, bounds["x"][1] - dx * 0.5, samples)
    ys = np.linspace(bounds["y"][0] + dy * 0.5, bounds["y"][1] - dy * 0.5, samples)
    zs = np.linspace(bounds["z"][0] + dz * 0.5, bounds["z"][1] - dz * 0.5, samples)
    points: list[tuple[float, float, float, float]] = []
    values: list[float] = []
    for x in xs:
        for y in ys:
            for z in zs:
                try:
                    value = evaluate_numeric_expression_vars(node, {"x": float(x), "y": float(y), "z": float(z)})
                except Exception:
                    value = float("nan")
                if math.isfinite(value):
                    points.append((float(x), float(y), float(z), float(value)))
                    values.append(float(value))
    if not points:
        raise ValueError("Could not sample this triple integral.")
    estimate = float(sum(values) * dx * dy * dz)
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0, z1 = bounds["z"]
    corners = [
        (x0, y0, z0, 0.0), (x1, y0, z0, 0.0), (x1, y1, z0, 0.0), (x0, y1, z0, 0.0),
        (x0, y0, z1, 0.0), (x1, y0, z1, 0.0), (x1, y1, z1, 0.0), (x0, y1, z1, 0.0),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return Model3DSpec("triple", "Triple Integral Volume", expression_text, corners + points, edges, x0, x1, y0, y1, z0, z1, estimate)
