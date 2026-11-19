"""Summarize scores only after all three independent deployments and their evaluations finish."""

import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("family", type=Path)
args = parser.parse_args()
baseline = json.loads((args.family / "frozen/frozen.json").read_text())
runs = [json.loads((args.family / f"run{i}/ttt.json").read_text()) for i in range(3)]
shift = baseline["config"].get("shift", "grid")
assert all(run["config"].get("shift", "grid") == shift for run in runs)
axis = list(range(0, 501, 50))
assert all([c["episode"] for c in run["checkpoints"]] == axis for run in runs)
assert all(run["encoder_sha"] == run["end_encoder_sha"] for run in runs)
assert all(
    run["initial_predictor_sha"]
    == baseline["initial_predictor_sha"]
    != run["last_predictor_sha"]
    for run in runs
)
for run in runs:
    for m in run["checkpoints"]:
        assert m["eval"]["episodes"]["seeds"] == baseline[shift]["episodes"]["seeds"]
        assert m["updates"] == m["nwindows"] // 64
scores = np.array([[c["eval"]["mean_end"] for c in r["checkpoints"]] for r in runs])
auc = np.trapz(scores, x=axis, axis=1) / 500
best = 1 + np.argmax(scores[:, 1:].mean(0))
pred = np.array([r["checkpoints"][-1]["five_block_mse"] for r in runs])
success = np.array([r["checkpoints"][-1]["eval"]["success_rate"] for r in runs])


def stats(x):
    return {
        "mean": float(np.mean(x)),
        "sample_sd": float(np.std(x, ddof=1)),
        "per_run": x.tolist(),
    }


report = {
    "frozen_original": baseline["original"]["mean_end"],
    f"frozen_{shift}": baseline[shift]["mean_end"],
    "frozen_mse": baseline["five_block_mse"],
    "frozen_success": baseline[shift]["success_rate"],
    "best_heldout_score": stats(scores[:, best]),
    "best_episode": axis[best],
    "auc": stats(auc),
    "final_heldout_score": stats(scores[:, -1]),
    "final_success": stats(success),
    "final_mse": stats(pred),
    "mse_percent_reduction": 100 * float(1 - pred.mean() / baseline["five_block_mse"]),
    "curves": scores.tolist(),
    "episode_axis": axis,
    "updates": [r["checkpoints"][-1]["updates"] for r in runs],
}
(args.family / "comparison.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
