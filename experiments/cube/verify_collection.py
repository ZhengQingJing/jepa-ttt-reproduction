"""Verify offline shapes, episode-level frame coverage and read-only array checksums."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/offline"))
    parser.add_argument("--output", type=Path, default=Path("collection-summary.json"))
    args = parser.parse_args()
    if not (args.data / "COMPLETE").is_file():
        raise ValueError("Collection is incomplete")
    rgb = np.load(args.data / "rgb.npy", mmap_mode="r")
    a = np.load(args.data / "action.npy", mmap_mode="r")
    p = np.load(args.data / "pos.npy", mmap_mode="r")
    assert rgb.shape == (3000, 50, 224, 224, 3) and rgb.dtype == np.uint8
    assert (
        a.shape == p.shape == (3000, 50, 3)
        and np.isfinite(a).all()
        and np.isfinite(p).all()
    )
    assert np.abs(a).max() <= 1.0
    for ep in range(len(rgb)):
        # Every raw observation contains a non-uniform camera image.
        assert np.all(np.ptp(rgb[ep], axis=(1, 2, 3)) > 0)
    hashes = {}
    for filename in ("rgb.npy", "action.npy", "pos.npy"):
        with (args.data / filename).open("rb") as fp:
            hashes[filename] = hashlib.file_digest(fp, "sha256").hexdigest()
    summary = {
        "episodes": len(rgb),
        "raw_observations": a.shape[0] * a.shape[1],
        "rgb_dtype": "uint8",
        "rgb_shape": list(rgb.shape),
        "action_shape": list(a.shape),
        "effector_shape": list(p.shape),
        "action_mean": a.reshape(-1, 3).mean(0, dtype=np.float64).tolist(),
        "action_std": a.reshape(-1, 3).std(0, ddof=1, dtype=np.float64).tolist(),
        "sha256": hashes,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
