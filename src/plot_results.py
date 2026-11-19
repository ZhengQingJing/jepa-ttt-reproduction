import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("family", type=Path)
args = parser.parse_args()
out = args.family
summary = json.loads((out / "comparison.json").read_text())
episodes = np.asarray(summary["episode_axis"])
runs = [json.loads((out / f"run{i}/ttt.json").read_text()) for i in range(3)]
shift = runs[0]["config"].get("shift", "grid")
scores = np.array(summary["curves"])
errors = np.array(
    [
        [summary["frozen_mse"]] + [c["five_block_mse"] for c in r["checkpoints"][1:]]
        for r in runs
    ]
)
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.25), layout="constrained")
for ax, data, ylabel in zip(
    axes, [scores, errors], ["Held-out planning score", "Five-block latent MSE"]
):
    mu, sd = data.mean(0), data.std(0, ddof=1)
    for values in data:
        ax.plot(episodes, values, color="#6a91ae", lw=0.85, alpha=0.62)
    ax.fill_between(
        episodes, mu - sd, mu + sd, color="#2b7396", alpha=0.15, label="1 sample SD"
    )
    ax.plot(
        episodes,
        mu,
        "o-",
        lw=2,
        color="#175675",
        markersize=3.5,
        label="TTT mean (3 runs)",
    )
    ax.axhline(data[0, 0], color="#ad592e", lw=1.6, ls="--", label="Frozen")
    ax.set(xlabel="Deployment episodes", ylabel=ylabel, xlim=(0, 500))
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylim(0, 0.86)
axes[0].legend(loc="lower right", fontsize=8, frameon=False)
axes[1].set_yscale("log")
fig.suptitle(
    "Two-Room grid rotation" if shift == "grid" else "Two-Room spatial wave",
    fontsize=14,
)
fig.savefig(out / "learning-curves.png", dpi=180)
