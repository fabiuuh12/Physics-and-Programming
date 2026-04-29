#!/usr/bin/env python3
"""Analyze risk-adaptive EKF parameter-scan results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rank_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    finite = [
        row
        for row in rows
        if np.isfinite(as_float(row, "rms_position_error"))
        and np.isfinite(as_float(row, "max_position_error"))
    ]
    return sorted(finite, key=lambda row: (as_float(row, "rms_position_error"), as_float(row, "max_position_error")))


def compare_to_baseline(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_run: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_run[row["run_name"]][row["filter"]] = row

    comparisons: list[dict[str, object]] = []
    for run_name, filters in sorted(by_run.items()):
        baseline = filters.get("baseline")
        if baseline is None:
            continue
        base_rms = as_float(baseline, "rms_position_error")
        for filter_name in ("adaptive_R", "adaptive_Q"):
            adaptive = filters.get(filter_name)
            if adaptive is None:
                continue
            rms = as_float(adaptive, "rms_position_error")
            improvement = 100.0 * (base_rms - rms) / base_rms if base_rms > 0.0 else 0.0
            comparisons.append(
                {
                    "run_name": run_name,
                    "filter": filter_name,
                    "baseline_rms_position_error": base_rms,
                    "adaptive_rms_position_error": rms,
                    "rms_improvement_percent": improvement,
                    "alpha": as_float(adaptive, "alpha"),
                    "beta": as_float(adaptive, "beta"),
                    "w_g": as_float(adaptive, "w_g"),
                    "adaptive_max_position_error": as_float(adaptive, "max_position_error"),
                    "adaptive_max_risk": as_float(adaptive, "max_risk"),
                }
            )
    return sorted(comparisons, key=lambda row: float(row["rms_improvement_percent"]), reverse=True)


def plot_best_filters(rows: list[dict[str, str]], path: Path) -> None:
    ranked = rank_rows(rows)[:12]
    labels = [f"{row['filter']}\n{row['run_name']}" for row in ranked]
    values = [as_float(row, "rms_position_error") for row in ranked]
    colors = [
        "tab:blue" if row["filter"] == "baseline" else "tab:green" if row["filter"] == "adaptive_Q" else "tab:orange"
        for row in ranked
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_ylabel("RMS position error [nondim]")
    ax.set_title("Best parameter-scan cases by RMS position error")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_improvement_heatmap(comparisons: list[dict[str, object]], filter_name: str, path: Path) -> None:
    selected = [row for row in comparisons if row["filter"] == filter_name]
    if not selected:
        return

    # If a filter ignores one parameter, average duplicate cells. This keeps the
    # plot truthful while preserving the full scan table.
    xs = sorted({float(row["beta"]) for row in selected})
    ys = sorted({float(row["w_g"]) for row in selected})
    grid = np.full((len(ys), len(xs)), np.nan)
    for iy, wg in enumerate(ys):
        for ix, beta in enumerate(xs):
            vals = [
                float(row["rms_improvement_percent"])
                for row in selected
                if float(row["w_g"]) == wg and float(row["beta"]) == beta
            ]
            if vals:
                grid[iy, ix] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(7, 4.8))
    im = ax.imshow(grid, origin="lower", cmap="RdYlGn", aspect="auto", vmin=-75, vmax=75)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([f"{value:g}" for value in xs])
    ax.set_yticks(np.arange(len(ys)))
    ax.set_yticklabels([f"{value:g}" for value in ys])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$w_g$")
    ax.set_title(f"{filter_name}: RMS improvement over baseline")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Improvement [%]")
    for iy in range(len(ys)):
        for ix in range(len(xs)):
            if np.isfinite(grid[iy, ix]):
                ax.text(ix, iy, f"{grid[iy, ix]:.0f}", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict[str, str]], comparisons: list[dict[str, object]]) -> None:
    ranked = rank_rows(rows)
    best = ranked[:5]
    worst = ranked[-5:]
    helpful = [row for row in comparisons if float(row["rms_improvement_percent"]) > 0.0]
    harmful = [row for row in comparisons if float(row["rms_improvement_percent"]) < 0.0]
    lines = [
        "# Parameter Scan Analysis",
        "",
        "This report is generated from `parameter_scan_summary.csv`.",
        "",
        "## Headline",
        "",
        f"- Total filter runs analyzed: {len(rows)}.",
        f"- Adaptive cases better than their same-run baseline: {len(helpful)}.",
        f"- Adaptive cases worse than their same-run baseline: {len(harmful)}.",
        "- This is an early controlled simulation, not a flight-performance claim.",
        "",
        "## Best RMS Position Error Cases",
        "",
        "| rank | filter | run | RMS position error | max position error | alpha | beta | w_g |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(best, start=1):
        lines.append(
            f"| {idx} | {row['filter']} | {row['run_name']} | "
            f"{as_float(row, 'rms_position_error'):.6g} | {as_float(row, 'max_position_error'):.6g} | "
            f"{as_float(row, 'alpha'):.3g} | {as_float(row, 'beta'):.3g} | {as_float(row, 'w_g'):.3g} |"
        )
    lines.extend(
        [
            "",
            "## Worst RMS Position Error Cases",
            "",
            "| rank | filter | run | RMS position error | max position error | alpha | beta | w_g |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(reversed(worst), start=1):
        lines.append(
            f"| {idx} | {row['filter']} | {row['run_name']} | "
            f"{as_float(row, 'rms_position_error'):.6g} | {as_float(row, 'max_position_error'):.6g} | "
            f"{as_float(row, 'alpha'):.3g} | {as_float(row, 'beta'):.3g} | {as_float(row, 'w_g'):.3g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The risk-aware estimator is useful only after parameter discipline.",
            "- Measurement-noise adaptation can help in stressed cases, but excessive inflation can also delay corrections.",
            "- Process-noise adaptation is a separate hypothesis and needs targeted tests before claiming superiority.",
            "- The next research step is controlled ablation: missed measurements only, geometry only, lighting only, then combined stress.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.summary)
    comparisons = compare_to_baseline(rows)
    ranked = rank_rows(rows)

    write_csv(args.out / "ranked_filters.csv", [dict(row) for row in ranked])
    write_csv(args.out / "baseline_comparisons.csv", comparisons)
    write_markdown(args.out / "scan_analysis.md", rows, comparisons)
    plot_best_filters(rows, args.out / "best_filters.png")
    plot_improvement_heatmap(comparisons, "adaptive_R", args.out / "adaptive_R_improvement_heatmap.png")
    plot_improvement_heatmap(comparisons, "adaptive_Q", args.out / "adaptive_Q_improvement_heatmap.png")

    summary = {
        "total_rows": len(rows),
        "best_filter": ranked[0] if ranked else None,
        "best_improvement": comparisons[0] if comparisons else None,
        "positive_improvement_cases": sum(1 for row in comparisons if float(row["rms_improvement_percent"]) > 0.0),
        "negative_improvement_cases": sum(1 for row in comparisons if float(row["rms_improvement_percent"]) < 0.0),
    }
    (args.out / "scan_analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
