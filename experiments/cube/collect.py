"""Collect unshifted native RGB clips under the paper's two exploration modes."""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from cube_env import GOAL, HORIZON, Hover


def field_specs(n):
    return {
        "rgb": ((n, HORIZON, 224, 224, 3), "uint8"),
        "action": ((n, HORIZON, 3), "float32"),
        "pos": ((n, HORIZON, 3), "float32"),
    }


def collect_shard(params):
    directory, n, shard, n_shards = params
    marker = directory / f"part-{shard}.done"
    if marker.is_file():
        return
    env = Hover()
    data = {k: np.load(directory / f"{k}.npy", mmap_mode="r+") for k in field_specs(n)}
    begin = time.monotonic()
    try:
        for ordinal, ep in enumerate(range(shard, n, n_shards), 1):
            pos = env.reset(100000 + ep)
            rng = np.random.default_rng(120000 + ep)
            for step in range(HORIZON):
                if ep % 2:
                    cmd = rng.normal(0.0, 0.65, 3)
                elif rng.random() < 0.30:
                    cmd = rng.uniform(-1.0, 1.0, 3)
                else:
                    cmd = (GOAL - pos) / 0.05 + rng.normal(0, 0.20, 3)
                cmd = np.clip(cmd, -1.0, 1.0).astype(np.float32)
                data["rgb"][ep, step] = env.render()
                data["pos"][ep, step] = pos
                data["action"][ep, step] = cmd
                pos, _, _ = env.step(cmd)
            if ordinal % 50 == 0:
                print(
                    f"worker {shard}: {ordinal} episodes, elapsed {time.monotonic()-begin:.1f}s",
                    flush=True,
                )
        for v in data.values():
            v.flush()
        marker.write_text("complete\n")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/offline"))
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    d = args.output
    d.mkdir(exist_ok=True, parents=True)
    specs = field_specs(args.episodes)
    contract = {
        "episodes": args.episodes,
        "workers": args.workers,
        "steps": HORIZON,
        "raw_observation": "224x224 three-channel native front view",
        "collection": "equal deterministic mixture of guided and Gaussian episodes",
        "guided": "30% independent uniform xyz; otherwise clipped goal displacement/0.05 plus Normal(0,0.20)",
        "gaussian": "independent clipped Normal(0,0.65) xyz",
        "reset_seed_base": 100000,
        "action_seed_base": 120000,
    }
    document = d / "protocol.json"
    if document.exists() and json.loads(document.read_text()) != contract:
        raise ValueError(
            "Collection configuration conflicts with the existing directory"
        )
    document.write_text(json.dumps(contract, indent=2) + "\n")
    for field, (shape, dtype) in specs.items():
        path = d / f"{field}.npy"
        if not path.exists():
            np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape).flush()
        arr = np.load(path, mmap_mode="r")
        if arr.shape != shape or arr.dtype != np.dtype(dtype):
            raise ValueError(f"Shape or dtype mismatch: {field}")
    jobs = [(d, args.episodes, i, args.workers) for i in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        pool.map(collect_shard, jobs)
    (d / "COMPLETE").write_text("complete\n")
    print(
        f"Complete: {args.episodes} episodes, {args.episodes*HORIZON} rows", flush=True
    )


if __name__ == "__main__":
    main()
