"""Verify action transforms, seeded resets, native images, and reachable hover geometry."""

import argparse
import json
from pathlib import Path

import numpy as np

from cube_env import GOAL, SHIFTS, Hover, transformed


def verify():
    cmd = np.asarray([0.9, -0.4, 0.6])
    np.testing.assert_allclose(
        transformed(cmd, np.zeros(3), "quadratic"), [0.18, 1.0, -0.48]
    )
    np.testing.assert_array_equal(transformed(cmd, np.zeros(3), "delayed"), np.zeros(3))
    np.testing.assert_array_equal(transformed(-cmd, cmd, "delayed"), cmd[[1, 2, 0]])
    results = {}
    for shift in SHIFTS:
        env = Hover(shift)
        try:
            pos = env.reset(51510)
            initial = pos.copy()
            rgb = env.render()
            assert rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
            # This geometric controller is only a task check. Axis moves null the
            # quadratic cross terms, while the delayed map uses inverse permutation.
            for t in range(1, 51):
                error = GOAL - pos
                k = np.argmax(np.abs(error))
                action = np.zeros(3)
                action[k] = np.clip(error[k] / 0.05, -1, 1)
                if shift == "delayed":
                    action = action[[2, 0, 1]]
                pos, score, success = env.step(action)
                if success:
                    break
            assert success
            results[shift] = {
                "seed": 51510,
                "start": initial.tolist(),
                "first_goal_step": t,
                "score": score,
                "distance_m": float(np.linalg.norm(pos - GOAL)),
                "rgb_shape": list(rgb.shape),
            }
            np.testing.assert_array_equal(env.reset(51510), initial)
            np.testing.assert_array_equal(env.previous, np.zeros(3))
        finally:
            env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("geometry.json"))
    args = parser.parse_args()
    result = verify()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
