"""Recompute the published planning and prediction metrics from the saved evaluations."""

import argparse
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_value(label, computed, reported):
    if not math.isclose(computed, reported, rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError(f"{label}: computed {computed}, file reports {reported}")


def check_shift(directory, shift):
    frozen = read(directory / "frozen" / "frozen.json")
    summary = read(directory / "comparison.json")
    if frozen["config"].get("shift", "grid") != shift or frozen["trajectories"] != 100:
        raise ValueError("Unexpected frozen shift or diagnostic suite size")
    fixed_seeds = frozen[shift]["episodes"]["seeds"]
    if len(set(fixed_seeds)) != 100:
        raise ValueError("Expected 100 distinct held-out seeds")
    base = statistics.mean(frozen[shift]["episodes"]["end_score"])
    check_value("Frozen score", base, summary["frozen_" + shift])
    check_value("Frozen MSE", frozen["five_block_mse"], summary["frozen_mse"])

    times = list(range(0, 501, 50))
    curves, areas, errors = [], [], []
    for run_id in range(3):
        run = read(directory / f"run{run_id}" / "ttt.json")
        if run["config"]["run"] != run_id or run["config"].get("shift", "grid") != shift:
            raise ValueError("Unexpected deployment seed or dynamics shift")
        checkpoints = run["checkpoints"]
        if [c["episode"] for c in checkpoints] != times:
            raise ValueError(f"Invalid evaluation times in run {run_id}")
        if (
            run["initial_predictor_sha"] != frozen["initial_predictor_sha"]
            or run["last_predictor_sha"] == run["initial_predictor_sha"]
            or run["encoder_sha"] != frozen["encoder_sha"]
            or run["end_encoder_sha"] != run["encoder_sha"]
        ):
            raise ValueError(f"Inconsistent model fingerprints in run {run_id}")
        curve = []
        for c in checkpoints:
            evaluation = c["eval"]["episodes"]
            if (
                evaluation["seeds"] != fixed_seeds
                or len(evaluation["end_score"]) != 100
            ):
                raise ValueError("Held-out suite mismatch")
            if (
                c["nwindows"] != c["episode"] * 32
                or c["updates"] != c["nwindows"] // 64
            ):
                raise ValueError("Dense replay window/update count mismatch")
            score = statistics.mean(evaluation["end_score"])
            check_value("Held-out mean", score, c["eval"]["mean_end"])
            curve.append(score)
        check_value("Episode-zero score", curve[0], base)
        area = (
            sum(
                (v + w) * (b - a) / 2
                for a, b, v, w in zip(times, times[1:], curve, curve[1:])
            )
            / times[-1]
        )
        curves.append(curve)
        areas.append(area)
        errors.append(checkpoints[-1]["five_block_mse"])
        final = checkpoints[-1]
        if final["trajectories"] != 100 or len(final["by_horizon"]) != 5:
            raise ValueError("Incomplete fixed-trajectory evaluation")
        check_value("Five-block MSE", statistics.mean(final["by_horizon"]), errors[-1])

    best_index = max(
        range(1, len(times)), key=lambda j: statistics.mean(c[j] for c in curves)
    )
    if times[best_index] != summary["best_episode"]:
        raise ValueError("Best checkpoint mismatch")
    groups = {
        "best_heldout_score": [c[best_index] for c in curves],
        "auc": areas,
        "final_mse": errors,
        "final_heldout_score": [c[-1] for c in curves],
    }
    for key, values in groups.items():
        check_value(f"{key} mean", statistics.mean(values), summary[key]["mean"])
        check_value(
            f"{key} sample SD", statistics.stdev(values), summary[key]["sample_sd"]
        )
    print(
        f"{shift}: frozen {base:.4f}, best {statistics.mean(groups['best_heldout_score']):.4f}"
        f" (episode {times[best_index]}), AUC {statistics.mean(areas):.4f},"
        f" five-block MSE {statistics.mean(errors):.5f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shifts", nargs="*", help="grid and/or wave (default: both)")
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parents[1] / "results"
    )
    args = parser.parse_args()
    for name in args.shifts or ("grid", "wave"):
        if name not in {"grid", "wave"}:
            parser.error(f"Unknown shift {name!r}")
        check_shift(args.results / f"two-room-{name}", name)
