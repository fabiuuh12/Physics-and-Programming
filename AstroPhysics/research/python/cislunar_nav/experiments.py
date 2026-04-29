"""Experiment orchestration for cislunar navigation simulations."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from .config import output_directory, save_config_snapshot
from .diagnostics import plot_run, save_run_npz, write_json, write_summary_csv
from .dynamics import propagate_truth
from .ekf import generate_measurements, run_filter, summarize_result


def run_experiment(config: dict[str, Any], run_name: str | None = None) -> list[dict[str, Any]]:
    out = output_directory(config, run_name)
    save_config_snapshot(config, out / "config_snapshot.json")
    dyn = config["dynamics"]
    truth = propagate_truth(
        np.array(config["initial"]["true_state"], dtype=float),
        int(dyn["steps"]),
        float(dyn["dt"]),
        float(dyn["mu"]),
    )
    measurements = generate_measurements(truth, config)
    results = [
        run_filter("baseline", truth, config, "baseline", measurements),
        run_filter("adaptive_R", truth, config, "adaptive_R", measurements),
        run_filter("adaptive_Q", truth, config, "adaptive_Q", measurements),
    ]
    rows = [summarize_result(result) for result in results]
    write_summary_csv(out / "summary.csv", rows)
    write_json(out / "summary.json", {"experiment": config["experiment_name"], "rows": rows})
    save_run_npz(out / "trajectories.npz", truth, measurements, results)
    plot_run(out / "diagnostics.png", truth, results)
    return rows


def run_parameter_scan(config: dict[str, Any]) -> list[dict[str, Any]]:
    scan = config["scan"]
    root = output_directory(config)
    all_rows: list[dict[str, Any]] = []
    for alpha in scan["alpha"]:
        for beta in scan["beta"]:
            for wg in scan["w_g"]:
                trial = copy.deepcopy(config)
                trial["filter"]["alpha"] = alpha
                trial["filter"]["beta"] = beta
                trial["risk"]["w_g"] = wg
                run_name = f"alpha_{alpha:g}_beta_{beta:g}_wg_{wg:g}".replace(".", "p")
                rows = run_experiment(trial, run_name)
                for row in rows:
                    row = dict(row)
                    row["run_name"] = run_name
                    row["alpha"] = alpha
                    row["beta"] = beta
                    row["w_g"] = wg
                    all_rows.append(row)
    write_summary_csv(root / "parameter_scan_summary.csv", all_rows)
    return all_rows
