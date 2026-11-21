"""Replay an archived Reacher command sequence in the native simulator."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reacher_task import Reacher, score, success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, help="Extracted directory containing replay.json and actions.npz")
    parser.add_argument("--frames", type=Path, help="Write native RGB frames into this directory")
    args = parser.parse_args()
    metadata = json.loads((args.clip / "replay.json").read_text(encoding="utf-8"))
    if args.frames:
        from PIL import Image

    with np.load(args.clip / "actions.npz", allow_pickle=False) as trace:
        for arm in ("frozen", "adapted"):
            commands = trace[f"{arm}_commands"]
            q = trace[f"{arm}_q"]
            result = metadata[arm]
            steps = result["steps"]
            if len(commands) != steps or len(q) != steps + 1:
                raise ValueError(f"Invalid trajectory length for {arm}")
            if args.frames:
                folder = args.frames / arm
                folder.mkdir(parents=True, exist_ok=True)
            env = Reacher()
            try:
                env.reset(metadata["selected_seed"])
                hit = False
                for t in range(steps + 1):
                    if t:
                        env.step(commands[t - 1], metadata["shift"])
                        hit |= success(env.position)
                    np.testing.assert_allclose(env.position, q[t], rtol=0, atol=1e-8)
                    if args.frames:
                        Image.fromarray(env.render()).save(folder / f"{t:03d}.png")
                np.testing.assert_allclose(score(env.position), result["score"], rtol=0, atol=1e-7)
                if hit != result["success"]:
                    raise ValueError(f"Goal-hit mismatch for {arm}")
                print(f"{arm}: verified {steps} actions, terminal score {result['score']:.6f}, hit={hit}")
            finally:
                env.close()


if __name__ == "__main__":
    main()
