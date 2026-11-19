"""Verify and aggregate three 500-episode deployments collected by src/benchmark.py."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def calculate(folder):
    def read(file):
        return json.loads((folder / file).read_text(encoding="utf-8"))

    base = read("frozen.json")
    cfg = base["settings"]
    if cfg["eval_n"] != 100 or base["trajectories"] != 100:
        raise ValueError(
            "Formal results require 100 held-out starts and 100 diagnostic episodes"
        )
    seeds = list(range(1000000, 1000100))
    frozen_scores = base["evaluations"][cfg["shift"]]["episodes"]
    if [r["seed"] for r in frozen_scores] != seeds:
        raise ValueError("Frozen seed suite mismatch")
    curves = []
    areas = []
    errors = []
    success = []
    for seed in range(3):
        run = read(f"run{seed}/ttt.json")
        if (
            run["run"] != seed
            or run["settings"] != cfg
            or run["final_predictor_sha"] == cfg["initial_predictor_sha"]
        ):
            raise ValueError("Model or seed mismatch")
        cps = run["checkpoints"]
        if [c["episode"] for c in cps] != list(range(0, 501, 50)):
            raise ValueError("Incomplete evaluation curve")
        curve = []
        for c in cps:
            ev = c["eval"]["episodes"]
            if (
                [v["seed"] for v in ev] != seeds
                or c["updates"] != c["nwindows"] // 64
                or c["nwindows"] != c["episode"] * (cfg["horizon"] - 18)
            ):
                raise ValueError("Held-out or update schedule mismatch")
            curve.append(mean(v["score"] for v in ev))
        if cps[-1]["trajectories"] != 100 or len(cps[-1]["by_horizon"]) != 5:
            raise ValueError("Incomplete trajectory diagnostic")
        areas.append(sum((a + b) * 0.05 for a, b in zip(curve, curve[1:])))
        curves.append(curve)
        errors.append(cps[-1]["five_block_mse"])
        success.append(cps[-1]["eval"]["success_rate"])
    best = max(range(1, 11), key=lambda i: mean(c[i] for c in curves))

    def aggregate(values):
        return {"mean": mean(values), "sample_sd": stdev(values), "per_run": values}

    return {
        "task": cfg["task"],
        "shift": cfg["shift"],
        "frozen_score": mean(v["score"] for v in frozen_scores),
        "best_episode": best * 50,
        "best": aggregate([c[best] for c in curves]),
        "auc": aggregate(areas),
        "frozen_mse": base["five_block_mse"],
        "final_mse": aggregate(errors),
        "mse_reduction_percent": 100 * (1 - mean(errors) / base["five_block_mse"]),
        "final_success": aggregate(success),
        "score_curves": curves,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="runs/<task>/<shift>")
    args = parser.parse_args()
    summary = calculate(args.directory)
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (args.directory / "comparison.json").write_text(text, encoding="utf-8")
    print(text)
