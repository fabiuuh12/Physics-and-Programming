#!/usr/bin/env python3
"""Summarize controlled scenario results across experiment folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def read_summary(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload["rows"]
    for row in rows:
        row["scenario"] = payload["experiment"]
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = ["scenario", *[key for key in rows[0].keys() if key != "scenario"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_by_scenario(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    scenarios = sorted({str(row["scenario"]) for row in rows})
    best_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        baseline = next(row for row in scenario_rows if row["filter"] == "baseline")
        best = min(scenario_rows, key=lambda row: float(row["rms_position_error"]))
        base_rms = float(baseline["rms_position_error"])
        best_rms = float(best["rms_position_error"])
        improvement = 100.0 * (base_rms - best_rms) / base_rms if base_rms > 0.0 else 0.0
        best_rows.append(
            {
                "scenario": scenario,
                "best_filter": best["filter"],
                "baseline_rms_position_error": base_rms,
                "best_rms_position_error": best_rms,
                "improvement_percent": improvement,
                "baseline_max_position_error": float(baseline["max_position_error"]),
                "best_max_position_error": float(best["max_position_error"]),
                "missed_fraction": float(best["missed_fraction"]),
            }
        )
    return best_rows


def plot_scenario_bars(rows: list[dict[str, object]], path: Path) -> None:
    scenarios = sorted({str(row["scenario"]) for row in rows})
    filters = ["baseline", "adaptive_R", "adaptive_Q"]
    x = np.arange(len(scenarios))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, filter_name in enumerate(filters):
        values = []
        for scenario in scenarios:
            match = next(row for row in rows if row["scenario"] == scenario and row["filter"] == filter_name)
            values.append(float(match["rms_position_error"]))
        ax.bar(x + (idx - 1) * width, values, width=width, label=filter_name)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", "\n") for name in scenarios], fontsize=9)
    ax.set_ylabel("RMS position error [nondim]")
    ax.set_title("Controlled scenario comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(path: Path, best_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Controlled Scenario Summary",
        "",
        "| scenario | best filter | baseline RMS | best RMS | improvement | notes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in best_rows:
        improvement = float(row["improvement_percent"])
        if improvement > 1.0:
            note = "adaptive filter improved RMS error"
        elif improvement < -1.0:
            note = "baseline remained better"
        else:
            note = "roughly tied with baseline"
        lines.append(
            f"| {row['scenario']} | {row['best_filter']} | "
            f"{float(row['baseline_rms_position_error']):.6g} | "
            f"{float(row['best_rms_position_error']):.6g} | "
            f"{improvement:.2f}% | {note} |"
        )
    lines.extend(
        [
            "",
            "These are first-pass deterministic scenario runs. The next step is Monte Carlo testing with multiple random seeds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary_paths = sorted(
        path
        for path in args.results_root.glob("*/summary.json")
        if "parameter_scan" not in path.parts
    )
    rows: list[dict[str, object]] = []
    for path in summary_paths:
        rows.extend(read_summary(path))
    best_rows = best_by_scenario(rows)
    write_csv(args.out / "scenario_all_filters.csv", rows)
    write_csv(args.out / "scenario_best_filters.csv", best_rows)
    write_markdown(args.out / "scenario_summary.md", best_rows)
    plot_scenario_bars(rows, args.out / "scenario_rms_comparison.png")


if __name__ == "__main__":
    main()
