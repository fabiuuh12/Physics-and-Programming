"""Diagnostics and output helpers for ASF experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def plot_field_grid(path: str | Path, fields: dict[str, np.ndarray]) -> None:
    """Plot a compact grid of scalar fields."""
    count = len(fields)
    cols = min(3, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, values) in zip(axes.ravel(), fields.items()):
        image = ax.imshow(values, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.axis("on")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def radial_profile(values: np.ndarray, dx: float, dy: float, bins: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Compute an approximate radial mean profile around the grid center."""
    ny, nx = values.shape
    yy, xx = np.indices(values.shape)
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)
    radius = np.sqrt(((xx - cx) * dx) ** 2 + ((yy - cy) * dy) ** 2)
    edges = np.linspace(0.0, float(radius.max()), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.zeros(bins, dtype=float)
    for i in range(bins):
        mask = (radius >= edges[i]) & (radius < edges[i + 1])
        profile[i] = float(np.mean(values[mask])) if np.any(mask) else np.nan
    return centers, profile


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)
