"""Output and plotting helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .ekf import FilterResult


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_run_npz(path: str | Path, truth: np.ndarray, measurements: np.ndarray, results: list[FilterResult]) -> None:
    arrays: dict[str, np.ndarray] = {"truth": truth, "measurements": measurements}
    for result in results:
        prefix = result.name
        arrays[f"{prefix}_estimates"] = result.estimates
        arrays[f"{prefix}_risks"] = result.risks
        arrays[f"{prefix}_errors"] = result.errors
        arrays[f"{prefix}_geometry"] = result.geometry
        arrays[f"{prefix}_missed"] = result.missed
        arrays[f"{prefix}_lighting"] = result.lighting
    np.savez_compressed(path, **arrays)


def plot_run(path: str | Path, truth: np.ndarray, results: list[FilterResult]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(truth[:, 0], truth[:, 1], "k-", label="truth")
    for result in results:
        ax.plot(result.estimates[:, 0], result.estimates[:, 1], label=result.name)
    ax.scatter([-0.0121505856, 0.9878494144], [0, 0], c=["tab:blue", "gray"], s=[80, 35], label="Earth/Moon")
    ax.set_title("Planar CR3BP trajectory")
    ax.set_xlabel("x [nondim]")
    ax.set_ylabel("y [nondim]")
    ax.axis("equal")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for result in results:
        ax.plot(np.linalg.norm(result.errors[:, :2], axis=1), label=result.name)
    ax.set_title("Position error")
    ax.set_xlabel("step")
    ax.set_ylabel("||delta r||")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for result in results:
        ax.plot(result.risks, label=result.name)
    ax.set_title("Navigation risk score")
    ax.set_xlabel("step")
    ax.set_ylabel("R_nav")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if results:
        ax.plot(results[0].geometry, label="geometry")
        ax.plot(results[0].missed, label="missed")
        ax.plot(results[0].lighting, label="lighting")
    ax.set_title("Stress drivers")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
