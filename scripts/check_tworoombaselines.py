"""Summarize a completed original-paper Two-Room ablation or episodic baseline."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def summarize(folder, mode):
    values = [json.loads((folder / f"run{i}.json").read_text()) for i in range(3)]

    def pair(x):
        return {"mean": mean(x), "sample_sd": stdev(x), "per_run": x}

    if mode == "episodic":
        scores = []
        for i, entry in enumerate(values):
            rows = entry["episodes"]
            if (
                entry["seed"] != i
                or len(rows) != 100
                or len({r["seed"] for r in rows}) != 100
            ):
                raise ValueError(
                    "Each seed must contain 100 distinct episode-local evaluations"
                )
            scores.append(mean(r["end_score"] for r in rows))
        return {"episodic_score": pair(scores), "episodes_per_run": 100}
    curves = []
    for i, entry in enumerate(values):
        points = entry["checkpoints"]
        if entry["seed"] != i or [c["episode"] for c in points] != list(
            range(0, 501, 50)
        ):
            raise ValueError("Expected complete 500-episode evaluation curve")
        for c in points:
            expected = (8 if entry["mode"] == "sparse" else 32) * c["episode"]
            if c["presentations"] != expected or c["updates"] != expected // 64:
                raise ValueError("Update schedule mismatch")
            if c["eval"]["episodes"]["seeds"] != list(range(1000000, 1000100)):
                raise ValueError("Held-out evaluation suite mismatch")
        curves.append([mean(c["eval"]["episodes"]["end_score"]) for c in points])
    best = max(range(1, 11), key=lambda t: mean(v[t] for v in curves))
    return {
        "best_episode": 50 * best,
        "best": pair([v[best] for v in curves]),
        "auc": pair([sum((a + b) * 0.05 for a, b in zip(v, v[1:])) for v in curves]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["episodic", "stream"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.mode), indent=2, allow_nan=False))
