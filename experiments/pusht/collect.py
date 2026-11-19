"""Collect compact RGB PushT episodes. Process one HDF5 shard per CPU worker."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import cv2
import h5py
import numpy as np
from task import PushTTask


def shard(job):
    out, first, stop, seed = job
    cv2.setNumThreads(0)
    dest = Path(out) / f"episodes-{first:04d}-{stop:04d}.h5"
    if dest.is_file():
        with h5py.File(dest) as f:
            if not (
                f.attrs["first"] == first
                and f.attrs["stop"] == stop
                and f.attrs["seed"] == seed
            ):
                raise ValueError(dest)
            return stop - first, int(f.attrs["rows"])
    temp = dest.with_suffix(".h5.partial")
    env = PushTTask()
    with h5py.File(temp, "w") as f:
        sizes = []
        for episode in range(first, stop):
            base_seed = seed + episode
            env.reset(base_seed)
            rng = np.random.default_rng(10000000 + base_seed)
            rgb = []
            cmd = []
            state = []
            score = []
            value = env.coverage()
            for t in range(300):
                if t % 4 == 0:
                    a = env.sample_command(rng)
                rgb.append(env.render())
                cmd.append(a.copy())
                state.append(env.state())
                score.append(value)
                value, hit = env.step(a)
                if hit:
                    break
            # Record the true final transition with one more image and no outgoing action.
            rgb.append(env.render())
            state.append(env.state())
            score.append(value)
            g = f.create_group(str(episode))
            g.create_dataset(
                "rgb", data=np.stack(rgb), chunks=(1, 224, 224, 3), compression="lzf"
            )
            g.create_dataset("action", data=np.stack(cmd))
            g.create_dataset("state", data=np.stack(state))
            g.create_dataset("coverage", data=np.asarray(score, np.float32))
            sizes.append(len(cmd))
        f.attrs.update(first=first, stop=stop, seed=seed, rows=sum(sizes))
    os.replace(temp, dest)
    print(f"completed {first}..{stop - 1}, {sum(sizes)} actions", flush=True)
    return stop - first, sum(sizes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--shard-size", type=int, default=25)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=100000)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    jobs = [
        (str(a.output), s, min(a.episodes, s + a.shard_size), a.seed)
        for s in range(0, a.episodes, a.shard_size)
    ]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        r = list(pool.map(shard, jobs))
    assert sum(x[0] for x in r) == a.episodes
    result = {
        "episodes": a.episodes,
        "actions": sum(x[1] for x in r),
        "rgb_frames": sum(x[1] + x[0] for x in r),
        "width": 224,
        "height": 224,
        "seed": a.seed,
        "shards": len(r),
    }
    (a.output / "manifest.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == "__main__":
    main()
