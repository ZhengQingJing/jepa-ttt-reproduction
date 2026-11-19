"""Fit a native-RGB action-conditioned LeWM and frozen score head for Reacher."""
import argparse
import json
import math
import time

import numpy as np
import torch
from reacher_task import DATA, EPISODES, MODELS, TRAIN_STEPS, score
from common import SIGReg, amp, encode, head_init, model_init
from experiment import write_json


def atomic_save(value, path):
    temp = path.with_suffix(".pending")
    torch.save(value, temp)
    temp.replace(path)


def load_data():
    meta = json.loads((DATA / "complete.json").read_text())
    assert meta["episodes"] == EPISODES and meta["raw_steps"] == TRAIN_STEPS
    x = np.load(DATA / "pixels.npy", mmap_mode="r").reshape(-1, 224, 224, 3)
    actions = np.asarray(np.load(DATA / "actions.npy", mmap_mode="r")).reshape(-1, 2)
    q = np.asarray(np.load(DATA / "qpos.npy", mmap_mode="r")).reshape(-1, 2)
    return x, actions, q


def rgb_tensor(frames):
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().cuda()


def train(model, pixels, raw_actions):
    starts = (np.arange(EPISODES)[:, None] * TRAIN_STEPS
              + np.arange(TRAIN_STEPS - 19)).reshape(-1)
    starts = np.random.default_rng(3072).permutation(starts)
    n = int(0.9 * len(starts))
    tr, va = starts[:n], starts[n:]
    actions = torch.tensor(raw_actions, device="cuda")
    actions = (actions - actions.mean(0)) / actions.std(0)
    delta = 5 * np.arange(4)
    ai = torch.arange(15, device="cuda")
    sigreg = SIGReg(17, 1024).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    nsteps = len(tr) // 128
    checkpoint = MODELS / "pretrain-state.pt"
    start_epoch = 0
    history = []
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        start_epoch = state["epoch"]
        history = state["history"]

    def objective(indices):
        image = rgb_tensor(pixels[(indices[:, None] + delta).reshape(-1)])
        ix = torch.as_tensor(indices, device="cuda")
        with amp():
            z = encode(model, image).reshape(len(ix), 4, 192)
            acts = actions[ix[:, None] + ai].reshape(len(ix), 3, 10)
            pred = model.predict(z[:, :3], model.action_encoder(acts))
        mse = (pred.float() - z[:, 1:].float()).square().mean()
        loss = mse + 0.09 * sigreg(z.float().transpose(0, 1))
        return loss, mse

    for ep in range(start_epoch, 20):
        model.train()
        perm = tr[torch.randperm(len(tr), device="cuda").cpu().numpy()]
        running = torch.zeros((), device="cuda")
        started = time.monotonic()
        for j in range(nsteps):
            t = ep * nsteps + j
            mult = (t + 1) / nsteps if ep == 0 else 0.5 * (
                1 + math.cos(math.pi * (t - nsteps) / (19 * nsteps))
            )
            opt.param_groups[0]["lr"] = 5e-5 * mult
            loss, mse = objective(perm[j * 128:(j + 1) * 128])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += mse.detach()
            if j == 0 or (j + 1) % 250 == 0:
                print("train", ep + 1, j + 1, nsteps, "mse", float(running / (j+1)),
                      "seconds", time.monotonic() - started, flush=True)
        model.eval()
        val = 0.0
        with torch.no_grad():
            for i in range(0, len(va), 128):
                batch = va[i:i + 128]
                val += float(objective(batch)[1]) * len(batch)
        history.append({"epoch":ep+1, "train_prediction_mse":float(running/nsteps),
                        "validation_prediction_mse":val/len(va), "optimizer_steps":nsteps*(ep+1)})
        atomic_save({"epoch": ep+1, "history":history, "model": model.state_dict(), "optimizer": opt.state_dict(),
                     "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()},
                    checkpoint)
        write_json(MODELS/"pretrain-metrics.json", {"train_windows":len(tr),"validation_windows":len(va),"epochs":history})
        print("epoch", ep+1, "validation_mse", val / len(va), flush=True)
    atomic_save(model.state_dict(), MODELS / "weights.pt")


def train_reward(model, pixels, actions, q):
    model.load_state_dict(torch.load(MODELS / "weights.pt", map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    torch.manual_seed(21017)
    y = torch.as_tensor(score(q), dtype=torch.float32, device="cuda")
    z = torch.empty((len(pixels), 192), device="cuda")
    with torch.no_grad(), amp():
        for i in range(0, len(z), 128):
            image = rgb_tensor(np.array(pixels[i:i+128], copy=True))
            z[i:i+128] = encode(model, image).float()
    perm = np.random.default_rng(21017).permutation(EPISODES)
    def rows(ids):
        return torch.from_numpy((ids[:, None] * TRAIN_STEPS + np.arange(TRAIN_STEPS)).reshape(-1)).cuda()
    train_ix = rows(perm[300:])
    val_ix = rows(perm[:300])
    head = head_init().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    best, stale, best_epoch = float("inf"), 0, 0
    for ep in range(1,101):
        head.train()
        for ix in train_ix[torch.randperm(len(train_ix), device="cuda")].split(512):
            loss = (head(z[ix]).flatten()-y[ix]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            mse = float((head(z[val_ix]).flatten()-y[val_ix]).square().mean())
        if mse < best:
            best, stale, best_epoch = mse, 0, ep
            atomic_save({"head":head.state_dict(), "epoch":ep, "val_mse":mse,
                         "mean":torch.from_numpy(actions.mean(0)), "std":torch.from_numpy(actions.std(0,ddof=1))},
                        MODELS / "reward.pt")
        else:
            stale += 1
        if ep % 5 == 0: print("reward", ep, mse, best, flush=True)
        if stale >= 15: break
    write_json(MODELS/"reward-metrics.json", {"selected_epoch":best_epoch,"validation_mse":best,
                                               "training_episodes":2700,"validation_episodes":300})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("world","reward"))
    args=p.parse_args()
    torch.manual_seed(3072)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    MODELS.mkdir(exist_ok=True,parents=True)
    model=model_init().cuda()
    pixels,action,q=load_data()
    if args.phase=="world": train(model,pixels,action)
    else: train_reward(model,pixels,action,q)
