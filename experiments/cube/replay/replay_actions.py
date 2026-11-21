"""Verify saved Cube commands, native states, scores and front-camera images."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cube_env import GOAL, Hover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, help="Extracted directory containing replay.json and two NPZ files")
    parser.add_argument("--frames", type=Path, help="Write native RGB frames into this directory")
    parser.add_argument("--check-pixels", action="store_true", help="Require exact agreement with archived RGB hashes")
    args = parser.parse_args()
    metadata = json.loads((args.clip / "replay.json").read_text(encoding="utf-8"))
    if args.frames:
        from PIL import Image

    for arm in ("frozen", "adapted"):
        result = metadata[arm]["result"]
        steps = result["steps"]
        if args.frames:
            folder = args.frames / arm
            folder.mkdir(parents=True, exist_ok=True)
        env = Hover(metadata["shift"])
        try:
            env.reset(metadata["evaluation_seed"])
            with np.load(args.clip / f"{arm}.npz", allow_pickle=False) as trace:
                np.testing.assert_array_equal(GOAL, trace["goal"])
                commands = trace["command"]
                states = trace["qpos"]
                xyz = trace["end_effector_xyz_m"]
                applied = trace["applied_command"]
                scores = trace["score"]
                hits = trace["hit"]
                hashes = trace["frame_sha256"]
                if len(commands) != steps or len(states) != steps + 1:
                    raise ValueError(f"Invalid trajectory length for {arm}")
                for t in range(steps + 1):
                    if t:
                        _, value, hit = env.step(commands[t - 1])
                        np.testing.assert_allclose(env.applied, applied[t - 1], rtol=0, atol=1e-8)
                        np.testing.assert_allclose(value, scores[t - 1], rtol=0, atol=1e-7)
                        if hit != hits[t - 1]:
                            raise ValueError(f"Goal-hit mismatch for {arm} at step {t}")
                    np.testing.assert_allclose(env.raw._data.qpos, states[t], rtol=0, atol=1e-7)
                    np.testing.assert_allclose(env.position, xyz[t], rtol=0, atol=1e-7)
                    if args.check_pixels or args.frames:
                        image = env.render()
                        if args.check_pixels and hashlib.sha256(image.tobytes()).hexdigest() != hashes[t]:
                            raise ValueError(f"RGB hash mismatch for {arm} at step {t}")
                        if args.frames:
                            Image.fromarray(image).save(folder / f"{t:03d}.png")
                np.testing.assert_allclose(scores[-1], result["score"], rtol=0, atol=1e-7)
                if bool(np.any(hits)) != result["success"]:
                    raise ValueError(f"Overall success mismatch for {arm}")
                print(f"{arm}: verified {steps} actions, terminal score {result['score']:.6f}, hit={result['success']}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
