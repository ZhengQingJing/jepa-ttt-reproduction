"""Pack native RGB and train the paper-size LeWM with PushT coverage supervision."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from paths import add_upstream_paths

add_upstream_paths()
import itertools

from common import SIGReg, amp, encode, head_init, model_init


def pack(source, dest):
    manifest = json.loads((source / "manifest.json").read_text())
    if (dest / "meta.npz").exists():
        return
    dest.mkdir(exist_ok=True, parents=True)
    frames = int(manifest["rgb_frames"])
    offsets = [0]
    actions = []
    scores = []
    temporary = dest / "frames.partial.npy"
    pixels = np.lib.format.open_memmap(
        temporary, mode="w+", shape=(frames, 224, 224, 3), dtype=np.uint8
    )
    for shard in sorted(source.glob("episodes-*.h5")):
        with h5py.File(shard) as f:
            for key in sorted(f, key=int):
                assert int(key) == len(offsets) - 1
                g = f[key]
                n = len(g["rgb"])
                assert n == len(g["action"]) + 1 == len(g["coverage"])
                begin = offsets[-1]
                pixels[begin : begin + n] = g["rgb"][:]
                actions.append(
                    np.concatenate((g["action"][:], np.zeros((1, 2), np.float32)))
                )
                scores.append(g["coverage"][:])
                offsets.append(begin + n)
    assert offsets[-1] == frames
    pixels.flush()
    del pixels
    os.replace(temporary, dest / "frames.npy")
    with (dest / "meta.partial.npz").open("wb") as out:
        np.savez(
            out,
            offsets=np.asarray(offsets, dtype=np.int64),
            action=np.concatenate(actions),
            score=np.concatenate(scores),
        )
    os.replace(dest / "meta.partial.npz", dest / "meta.npz")
    print("packed", manifest, flush=True)


def save(path, data):
    temporary = path.with_suffix(".partial.pt")
    torch.save(data, temporary)
    os.replace(temporary, path)


def pretrain(model, data, images, out, epochs):
    offsets = data["offsets"]
    starts = np.concatenate(
        [
            np.arange(s, e - 19, dtype=np.int64)
            for s, e in itertools.pairwise(offsets)
            if e - s >= 20
        ]
    )
    rng = np.random.default_rng(3072)
    split = int(0.9 * len(starts))
    starts = starts[rng.permutation(len(starts))]
    tr = starts[:split]
    va = starts[split:]
    act = data["action"]
    valid = np.ones(len(act), bool)
    valid[offsets[1:] - 1] = False
    mean = act[valid].mean(0)
    std = act[valid].std(0, ddof=1)
    a = torch.from_numpy((act - mean) / std).cuda()
    regularizer = SIGReg(knots=17, num_proj=1024).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.001)
    n_steps = len(tr) // 128
    ctx = np.arange(4) * 5
    action_ix = torch.arange(15, device="cuda")
    assert n_steps > 0
    first = 0
    resume = out / "pretrain-state.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["epochs"] != epochs:
            raise ValueError("Checkpoint schedule mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        rng.bit_generator.state = json.loads(state["rng"])
        first = state["completed"]
        del state

    def objective(indices):
        im = (
            torch.from_numpy(images[(indices[:, None] + ctx).reshape(-1)])
            .permute(0, 3, 1, 2)
            .cuda()
        )
        ix = torch.as_tensor(indices, device="cuda")
        with amp():
            z = encode(model, im).reshape(len(ix), 4, 192)
            a0 = a[ix[:, None] + action_ix].reshape(len(ix), 3, 10)
            pred = model.predict(z[:, :3], model.action_encoder(a0))
        mse = (pred.float() - z[:, 1:].float()).square().mean()
        return mse + 0.09 * regularizer(z.float().transpose(0, 1)), mse

    for ep in range(first, epochs):
        model.train()
        perm = tr[rng.permutation(len(tr))]
        st = time.monotonic()
        total = torch.zeros((), device="cuda")
        for j in range(n_steps):
            global_step = ep * n_steps + j
            mult = (
                (j + 1) / n_steps
                if ep == 0
                else 0.5
                * (
                    1
                    + math.cos(
                        math.pi
                        * (global_step - n_steps)
                        / max(1, (epochs - 1) * n_steps)
                    )
                )
            )
            optimizer.param_groups[0]["lr"] = 5e-5 * mult
            optimizer.zero_grad(set_to_none=True)
            loss, mse = objective(perm[j * 128 : (j + 1) * 128])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += mse.detach()
            if j in (0, 9) or (j + 1) % 200 == 0:
                print(
                    "train",
                    ep + 1,
                    j + 1,
                    n_steps,
                    "mse",
                    float(total / (j + 1)),
                    "seconds",
                    round(time.monotonic() - st, 1),
                    flush=True,
                )
        model.eval()
        v = 0.0
        with torch.no_grad():
            for i in range(0, len(va), 128):
                ids = va[i : i + 128]
                _, m = objective(ids)
                v += m.item() * len(ids)
        save(
            resume,
            {
                "epochs": epochs,
                "completed": ep + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "rng": json.dumps(rng.bit_generator.state),
                "val_predict_mse": v / len(va),
            },
        )
        print("epoch", ep + 1, "val_predict_mse", v / len(va), flush=True)
    save(out / "world.pt", model.state_dict())
    save(
        out / "action-stats.pt",
        {"mean": torch.from_numpy(mean), "std": torch.from_numpy(std)},
    )


def train_head(model, data, images, out):
    model.load_state_dict(
        torch.load(out / "world.pt", weights_only=True, map_location="cpu")
    )
    model.eval().requires_grad_(False)
    torch.manual_seed(21017)
    z = torch.empty((len(images), 192), device="cuda")
    with torch.no_grad(), amp():
        for i in range(0, len(images), 128):
            im = (
                torch.from_numpy(np.array(images[i : i + 128]))
                .permute(0, 3, 1, 2)
                .cuda()
            )
            z[i : i + len(im)] = encode(model, im).float()
    offsets = data["offsets"]
    perm = np.random.default_rng(21017).permutation(len(offsets) - 1)
    cut = max(1, len(perm) // 10)

    def index(episodes):
        return torch.from_numpy(
            np.concatenate([np.arange(offsets[e], offsets[e + 1]) for e in episodes])
        ).cuda()

    tr = index(perm[cut:])
    va = index(perm[:cut])
    y = torch.from_numpy(data["score"]).cuda()
    head = head_init().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=1e-5)
    best = math.inf
    stale = 0
    for ep in range(100):
        head.train()
        for idx in tr[torch.randperm(len(tr), device="cuda")].split(512):
            loss = (head(z[idx]).squeeze(-1) - y[idx]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            head.eval()
            err = sum(
                float((head(z[i]).squeeze(-1) - y[i]).square().sum())
                for i in va.split(4096)
            ) / len(va)
        if err < best:
            best = err
            stale = 0
            stats = torch.load(out / "action-stats.pt", weights_only=True)
            save(
                out / "reward.pt",
                dict(head=head.state_dict(), val_mse=best, epoch=ep + 1, **stats),
            )
        else:
            stale += 1
        print("head", ep + 1, "mse", err, flush=True)
        if stale >= 15:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["pack", "train", "head"])
    p.add_argument("--source", type=Path, default=Path("data/offline"))
    p.add_argument("--packed", type=Path, default=Path("data/packed"))
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()
    if args.stage == "pack":
        pack(args.source, args.packed)
        return
    torch.manual_seed(3072)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    images = np.load(args.packed / "frames.npy", mmap_mode="r")
    data = np.load(args.packed / "meta.npz")
    args.models.mkdir(parents=True, exist_ok=True)
    model = model_init().cuda()
    if args.stage == "train":
        pretrain(model, data, images, args.models, args.epochs)
    else:
        train_head(model, data, images, args.models)


if __name__ == "__main__":
    main()
