"""Exercise both PushT interventions and replay selected completed offline episodes."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from task import PushTTask


def verify_shifts():
    plain = PushTTask()
    contact = PushTTask("contact")
    rot = PushTTask("action")
    for env in (plain, contact, rot):
        env.reset(101)
    np.testing.assert_array_equal(plain.state(), contact.state())
    np.testing.assert_array_equal(plain.render(), contact.render())
    center = np.asarray(rot.base.agent.position)
    command = (center + [20.0, 0.0]) / 256.0 - 1.0
    np.testing.assert_allclose(rot.target(command), center + [0.0, 20.0])
    stationary = np.asarray(plain.base.agent.position) / 256.0 - 1.0
    for _ in range(3):
        plain.step(stationary)
        contact.step(stationary)
    np.testing.assert_array_equal(plain.state(), contact.state())
    rng = np.random.default_rng(3818)
    target = unary_union([Polygon(poly) for poly in plain.targets])
    geom_error = 0.0
    for t in range(300):
        if t % 4 == 0:
            cmd = contact.sample_command(rng)
        value, _ = contact.step(cmd)
        if t % 10 == 0:
            block = unary_union(
                [Polygon(poly) for poly in contact.polys(contact.base.block)]
            )
            geom_error = max(
                geom_error, abs(value - target.intersection(block).area / target.area)
            )
    assert geom_error < 1e-5
    assert (
        contact.moving_contact_substeps > 0
        and contact.max_contact_rotation_error < 1e-9
    )
    assert contact.max_free_rotation == 0.0
    plain.base.block.angle = plain.base.goal_pose[2]
    plain.base.block.position = tuple(plain.base.goal_pose[:2])
    assert abs(plain.coverage() - 1.0) < 1e-6
    result = {
        "rgb_shape": list(plain.render().shape),
        "action_rotation_degrees": 90,
        "perfect_overlap": plain.coverage(),
        "contact_or_release_substeps": contact.substep_contact_pairs,
        "nonzero_contact_translations": contact.moving_contact_substeps,
        "maximum_contact_translation_rotation_residual": contact.max_contact_rotation_error,
        "maximum_free_substep_delta_change": contact.max_free_rotation,
        "maximum_sampled_coverage_difference_vs_shapely": geom_error,
    }
    for env in (plain, contact, rot):
        env.close()
    return result


def verify_dataset(root):
    manifest = json.loads((root / "manifest.json").read_text())
    shards = sorted(root.glob("episodes-*.h5"))
    count = steps = 0
    val_episodes = set(
        np.random.default_rng(21017).permutation(manifest["episodes"])[
            : manifest["episodes"] // 10
        ]
    )
    val_scores = []
    for file in shards:
        with h5py.File(file) as f:
            assert sorted(map(int, f)) == list(range(f.attrs["first"], f.attrs["stop"]))
            for key in f:
                e = f[key]
                n = len(e["action"])
                assert (
                    0 < n <= 300
                    and e["rgb"].shape == (n + 1, 224, 224, 3)
                    and e["rgb"].dtype == np.uint8
                )
                assert e["coverage"].shape == (n + 1,) and e["state"].shape == (
                    n + 1,
                    7,
                )
                if int(key) in val_episodes:
                    val_scores.append(e["coverage"][:])
                count += 1
                steps += n
    assert manifest["episodes"] == count == 3000 and manifest["actions"] == steps
    verified = []
    for file in (shards[0], shards[len(shards) // 2], shards[-1]):
        with h5py.File(file) as f:
            episode = int(f.attrs["first"])
            g = f[str(episode)]
            frames = g["rgb"][:]
            states = g["state"][:]
            commands = g["action"][:]
            y = g["coverage"][:]
            env = PushTTask()
            env.reset(episode + int(f.attrs["seed"]))
            rng = np.random.default_rng(10000000 + episode + int(f.attrs["seed"]))
            for t in range(len(commands) + 1):
                np.testing.assert_array_equal(env.render(), frames[t])
                np.testing.assert_allclose(env.state(), states[t], rtol=0.0, atol=1e-8)
                assert abs(env.coverage() - y[t]) < 1e-6
                if t < len(commands):
                    if t % 4 == 0:
                        cmd = env.sample_command(rng)
                    np.testing.assert_array_equal(cmd, commands[t])
                    env.step(cmd)
            env.close()
            verified.append(
                {
                    "episode": episode,
                    "raw_actions": len(commands),
                    "bit_exact_rgb_frames": len(frames),
                }
            )
    scores = np.concatenate(val_scores)
    return {
        **manifest,
        "verified_replay": verified,
        "reward_validation": {
            "frames": len(scores),
            "below_0.1": int((scores < 0.1).sum()),
            "at_least_0.5": int((scores >= 0.5).sum()),
            "max": float(scores.max()),
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = {"environment": verify_shifts()}
    if args.data is not None:
        result["offline_dataset"] = verify_dataset(args.data)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
