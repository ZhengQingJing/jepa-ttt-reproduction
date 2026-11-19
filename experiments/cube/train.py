"""Twenty-epoch LeWM pretraining and an episode-disjoint offline hover reward head."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from cube_env import task_score
from cube_model import SIGReg, amp, encode, head_init, make_model


def dataset(path):
    if not (path / "COMPLETE").exists():
        raise ValueError("Data collection must finish before training.")
    rgb = np.load(path / "rgb.npy", mmap_mode="r")
    actions = np.load(path / "action.npy", mmap_mode="r")
    positions = np.load(path / "pos.npy", mmap_mode="r")
    assert rgb.shape[:2] == actions.shape[:2] == positions.shape[:2] == (3000, 50)
    assert rgb.shape[2:] == (224, 224, 3)
    return (
        rgb.reshape(-1, 224, 224, 3),
        actions.reshape(-1, 3),
        positions.reshape(-1, 3),
    )


def rgb_tensor(rgb, idx):
    # Advanced indexing gathers only this batch from the memory-mapped native frames.
    return torch.from_numpy(rgb[idx]).permute(0, 3, 1, 2).to("cuda")


def atomic_checkpoint(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def pretrain(model, rgb, actions, models):
    starts = (np.arange(3000)[:, None] * 50 + np.arange(31)).ravel()
    starts = np.random.default_rng(3072).permutation(starts)
    cut = int(0.9 * len(starts))
    train = starts[:cut]
    val = starts[cut:]
    a = torch.from_numpy(np.array(actions)).to("cuda")
    a = (a - a.mean(0)) / a.std(0)
    keyframes = np.arange(4) * 5
    action_indices = torch.arange(15, device="cuda")
    reg = SIGReg(17, 1024).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    steps = len(train) // 128
    epoch_start = 0
    progress = []
    resume = models / "training-state.pt"
    if resume.is_file():
        old = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(old["model"])
        opt.load_state_dict(old["optimizer"])
        torch.set_rng_state(old["rng"])
        torch.cuda.set_rng_state(old["cuda_rng"])
        progress = old["history"]
        epoch_start = old["epoch"]

    def objective(idx, training):
        px = rgb_tensor(rgb, (idx[:, None] + keyframes).ravel())
        act = a[torch.from_numpy(idx).to("cuda")[:, None] + action_indices].view(
            len(idx), 3, 15
        )
        with amp():
            z = encode(model, px).reshape(len(idx), 4, 192)
            pred = model.predict(z[:, :3], model.action_encoder(act))
        mse = (pred.float() - z[:, 1:].float()).square().mean()
        return (mse + 0.09 * reg(z.float().transpose(0, 1)) if training else mse), mse

    for ep in range(epoch_start, 20):
        now = time.monotonic()
        model.train()
        order = train[torch.randperm(len(train), device="cuda").cpu().numpy()]
        total = torch.zeros((), device="cuda")
        for j in range(steps):
            step = ep * steps + j
            mult = (
                (step + 1) / steps
                if ep == 0
                else 0.5 * (1 + math.cos(math.pi * (step - steps) / (19 * steps)))
            )
            opt.param_groups[0]["lr"] = 5e-5 * mult
            loss, mse = objective(order[j * 128 : (j + 1) * 128], True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += mse.detach()
            if j == 0 or (j + 1) % 100 == 0:
                print(
                    "pretraining",
                    ep + 1,
                    j + 1,
                    steps,
                    "mse",
                    float(total / (j + 1)),
                    "peak_gib",
                    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                    "seconds",
                    round(time.monotonic() - now, 1),
                    flush=True,
                )
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for j in range(0, len(val), 128):
                idx = val[j : j + 128]
                _, mse = objective(idx, False)
                val_sum += float(mse) * len(idx)
        progress.append(
            {
                "epoch": ep + 1,
                "steps": steps,
                "train_mse": float(total / steps),
                "val_mse": val_sum / len(val),
            }
        )
        print("epoch complete", progress[-1], flush=True)
        atomic_checkpoint(
            {
                "epoch": ep + 1,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "history": progress,
            },
            resume,
        )
        (models / "pretraining.json").write_text(json.dumps(progress, indent=2) + "\n")
    atomic_checkpoint(model.state_dict(), models / "cube-weights.pt")


def reward(model, rgb, actions, positions, models):
    torch.manual_seed(21017)
    model.eval().requires_grad_(False)
    labels = torch.from_numpy(task_score(positions).astype(np.float32)).cuda()
    features = []
    with torch.no_grad(), amp():
        for i in range(0, len(rgb), 128):
            ix = np.arange(i, min(len(rgb), i + 128))
            features.append(encode(model, rgb_tensor(rgb, ix)).float())
    z = torch.cat(features)
    perm = np.random.default_rng(21017).permutation(3000)
    rows = lambda eps: torch.from_numpy((eps[:, None] * 50 + np.arange(50)).ravel()).to(
        "cuda"
    )
    train, val = rows(perm[300:]), rows(perm[:300])
    head = head_init().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    best, stale = math.inf, 0
    for ep in range(1, 101):
        head.train()
        for idx in train[torch.randperm(len(train), device="cuda")].split(512):
            mse = (head(z[idx]).flatten() - labels[idx]).square().mean()
            opt.zero_grad(set_to_none=True)
            mse.backward()
            opt.step()
        with torch.no_grad():
            err = float((head.eval()(z[val]).flatten() - labels[val]).square().mean())
        if err < best:
            best, stale = err, 0
            atomic_checkpoint(
                {
                    "head": head.state_dict(),
                    "epoch": ep,
                    "val_mse": best,
                    "mean": torch.from_numpy(actions.mean(0)),
                    "std": torch.from_numpy(actions.std(0, ddof=1)),
                },
                models / "cube-reward.pt",
            )
        else:
            stale += 1
        if ep % 10 == 0:
            print("head", ep, err, "best", best, flush=True)
        if stale == 15:
            break
    print("reward complete", best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["pretrain", "reward"])
    p.add_argument("--data", type=Path, default=Path("data/offline"))
    p.add_argument("--models", type=Path, default=Path("models"))
    args = p.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(3072)
    model = make_model().cuda()
    rgb, actions, pos = dataset(args.data)
    if args.phase == "pretrain":
        pretrain(model, rgb, actions, args.models)
    else:
        model.load_state_dict(
            torch.load(
                args.models / "cube-weights.pt", map_location="cpu", weights_only=True
            )
        )
        reward(model, rgb, actions, pos, args.models)
