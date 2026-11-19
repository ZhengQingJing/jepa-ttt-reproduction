import argparse
import math
import time
import numpy as np
import torch
from common import (
    ROOT,
    TwoRoomEnv,
    OfflineRenderer,
    SIGReg,
    amp,
    encode,
    head_init,
    model_init,
    score,
)


def weights(model, path):
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=True
    )


def reward_model(model, renderer, data, name):
    torch.manual_seed(21017)
    model.requires_grad_(False).eval()
    p = torch.as_tensor(data["pos"], device="cuda")
    y = torch.as_tensor(score(data["pos"]), device="cuda", dtype=torch.float32)
    t0 = time.monotonic()
    with torch.no_grad(), amp():
        chunks = []
        for start in range(0, len(p), 128):
            chunks.append(encode(model, renderer(p[start : start + 128])).float())
        z = torch.cat(chunks)
    print(f"features {len(z)} in {time.monotonic() - t0:.1f}s", flush=True)
    off = data["offsets"]
    perm = np.random.default_rng(21017).permutation(len(off) - 1)

    def rows(eps):
        return torch.as_tensor(
            np.concatenate([np.arange(off[i], off[i + 1]) for i in eps]), device="cuda"
        )

    train, val = rows(perm[300:]), rows(perm[:300])
    head = head_init().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.00001)
    best, stale = math.inf, 0
    for epoch in range(1, 101):
        head.train()
        for ix in train[torch.randperm(len(train), device="cuda")].split(512):
            err = (head(z[ix]).flatten() - y[ix]).square().mean()
            opt.zero_grad(set_to_none=True)
            err.backward()
            opt.step()
        with torch.no_grad():
            head.eval()
            h = head(z[val]).flatten()
            err = (h - y[val]).square().mean().item()
        if err < best:
            best, stale = err, 0
            torch.save(
                {
                    "head": head.state_dict(),
                    "val_mse": best,
                    "epoch": epoch,
                    "mean": torch.from_numpy(data["action"].mean(0)),
                    "std": torch.from_numpy(data["action"].std(0, ddof=1)),
                },
                ROOT / f"models/{name}-reward.pt",
            )
        else:
            stale += 1
        if epoch % 10 == 0:
            print("head", epoch, err, "best", best, flush=True)
        if stale >= 15:
            break
    print("reward fit done", best, flush=True)


def pretrain(model, renderer, data):
    # Match the upstream dataset's four rows with a frame skip of five.
    starts = np.concatenate(
        [
            np.arange(a, b - 19, dtype=np.int64)
            for a, b in zip(data["offsets"][:-1], data["offsets"][1:])
            if b - a >= 20
        ]
    )
    rng = np.random.default_rng(3072)
    starts = rng.permutation(starts)
    n = int(0.9 * len(starts))
    tr = torch.as_tensor(starts[:n], device="cuda")
    va = torch.as_tensor(starts[n:], device="cuda")
    pos = torch.tensor(data["pos"], device="cuda")
    a = torch.tensor(data["action"], device="cuda")
    a = (a - a.mean(0)) / a.std(0)
    delta = torch.arange(4, device="cuda") * 5
    act_ix = torch.arange(15, device="cuda")
    reg = SIGReg(17, 1024).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

    def objective(ix):
        frames = renderer(pos[ix[:, None] + delta].reshape(-1, 2))
        with amp():
            z = encode(model, frames).reshape(len(ix), 4, 192)
            acts = a[ix[:, None] + act_ix].view(len(ix), 3, 10)
            pred = model.predict(z[:, :3], model.action_encoder(acts))
        mse = (pred.float() - z[:, 1:].float()).square().mean()
        return mse + 0.09 * reg(z.float().transpose(0, 1)), mse

    # The paper specifies linear warm-up then cosine decay but omits the duration.
    steps = len(tr) // 128
    for ep in range(20):
        model.train()
        perm = tr[torch.randperm(len(tr), device="cuda")]
        total = torch.zeros((), device="cuda")
        now = time.monotonic()
        for j in range(steps):
            step = ep * steps + j
            mult = (
                (step + 1) / steps
                if ep == 0
                else 0.5 * (1 + math.cos(math.pi * (step - steps) / (19 * steps)))
            )
            optimizer.param_groups[0]["lr"] = 5e-5 * mult
            loss, mse = objective(perm[j * 128 : (j + 1) * 128])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += mse.detach()
            if j == 0 or (j + 1) % 200 == 0:
                print(
                    "pretrain",
                    ep + 1,
                    j + 1,
                    steps,
                    "mse",
                    round(float(total / (j + 1)), 5),
                    "sec",
                    round(time.monotonic() - now, 1),
                    flush=True,
                )
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for ix in va.split(128):
                _, mse = objective(ix)
                val_sum += mse.item() * len(ix)
        print("epoch", ep + 1, "val_pred_mse", val_sum / len(va), flush=True)
        torch.save(
            {"epoch": ep + 1, "val_pred_mse": val_sum / len(va)},
            ROOT / "logs/scratch-progress.pt",
        )
    torch.save(model.state_dict(), ROOT / "models/scratch-weights.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scratch-train", "scratch-head"])
    args = ap.parse_args()
    torch.manual_seed(3072)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    model = model_init().cuda()
    data = np.load(ROOT / "data/offline.npz")
    env = TwoRoomEnv()
    env.reset(seed=17)
    renderer = OfflineRenderer(env)
    if args.mode == "scratch-train":
        pretrain(model, renderer, data)
    else:
        name = "scratch"
        weights(model, ROOT / f"models/{name}-weights.pt")
        reward_model(model, renderer, data, name)
