"""Run the frozen control, three independent dense-replay deployments and aggregation."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def execute(script, *args):
    subprocess.run(
        [sys.executable, str(ROOT / "src" / script), *map(str, args)],
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shift", choices=["grid", "wave"])
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="train a new world model and reward head before planning",
    )
    args = parser.parse_args()
    for name in ["data", "models", "logs", "runs"]:
        (ROOT / name).mkdir(exist_ok=True)
    if args.from_scratch:
        if not (ROOT / "data/offline.npz").exists():
            execute("collect.py")
        execute("offline.py", "scratch-train")
        execute("offline.py", "scratch-head")
    settings = ["--shift", args.shift, "--base", "scratch", "--chunk", "4"]
    execute("experiment.py", "frozen", *settings, "--out", f"{args.shift}/frozen")
    for seed in range(3):
        execute(
            "experiment.py",
            "ttt",
            *settings,
            "--run",
            seed,
            "--out",
            f"{args.shift}/run{seed}",
        )
    family = ROOT / "runs" / args.shift
    execute("summarize.py", family)
    execute("plot_results.py", family)


if __name__ == "__main__":
    main()
