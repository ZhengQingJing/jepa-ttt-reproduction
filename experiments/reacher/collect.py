"""Collect 3,000 original-dynamics random-action episodes and native RGB keyframes."""
import argparse
import json
import multiprocessing as mp
import time

import numpy as np
from numpy.lib.format import open_memmap

from reacher_task import DATA, EPISODES, TRAIN_STEPS, Reacher


SHAPES = {
    "pixels": ((EPISODES, TRAIN_STEPS, 224, 224, 3), np.uint8),
    "qpos": ((EPISODES, TRAIN_STEPS, 2), np.float32),
    "actions": ((EPISODES, TRAIN_STEPS, 2), np.float32),
}
SHARD_SIZE = 50


def collect_shard(shard):
    marker = DATA / f"shard-{shard:03}.complete"
    if marker.is_file():
        return shard
    env = Reacher()
    outputs = {k: np.load(DATA / f"{k}.npy", mmap_mode="r+") for k in SHAPES}
    try:
        for episode in range(shard * SHARD_SIZE, min((shard + 1) * SHARD_SIZE, EPISODES)):
            env.reset(100000 + episode)
            rng = np.random.default_rng(130000 + episode)
            for step in range(TRAIN_STEPS):
                command = rng.uniform(-1, 1, 2).astype(np.float32)
                outputs["qpos"][episode, step] = env.position
                outputs["pixels"][episode, step] = env.render()
                outputs["actions"][episode, step] = command
                env.step(command, "original")
        for array in outputs.values():
            array.flush()
        marker.write_text("complete\n")
    finally:
        env.close()
    return shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    for key, (shape, dtype) in SHAPES.items():
        path = DATA / f"{key}.npy"
        if not path.exists():
            output = open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            del output
        else:
            array = np.load(path, mmap_mode="r")
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"Invalid array: {path}")
    started = time.monotonic()
    count = (EPISODES + SHARD_SIZE - 1) // SHARD_SIZE
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, shard in enumerate(pool.imap_unordered(collect_shard, range(count)), 1):
            print(f"completed {n}/{count} shards; shard={shard}, elapsed={time.monotonic()-started:.0f}s", flush=True)
    assert all((DATA / f"shard-{i:03}.complete").exists() for i in range(count))
    metadata = {"episodes": EPISODES, "raw_steps": TRAIN_STEPS, "rows": EPISODES * TRAIN_STEPS,
                "native_rgb": [224, 224, 3], "policy": "uniform random [-1, 1]^2",
                "environment_seed": "100000+episode", "command_seed": "130000+episode",
                "raw_action_repeat": 2, "target": [0.0, 0.0]}
    (DATA / "complete.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
