import argparse
import hashlib
import json
import time
from collections import deque
import numpy as np
import torch
from common import (
    ROOT,
    amp,
    encode,
    grid_action,
    wave_action,
    head_init,
    make_eval_env,
    model_init,
    score,
)
from offline import weights


def digest(module, include):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        if name.split(".")[0] in include:
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(".pending")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


class Planner:
    def __init__(self, model, head, mean, std):
        self.model, self.head = model, head
        self.mean, self.std = mean.cuda(), std.cuda()
        self.action_dim = self.mean.numel()
        self.block_dim = 5 * self.action_dim

    def action_embeddings(self, blocks):
        b, t = blocks.shape[:2]
        flattened = ((blocks.reshape(b, t, 5, self.action_dim) - self.mean) / self.std).reshape(
            b, t, self.block_dim
        )
        return self.model.action_encoder(flattened)

    def rollout(self, z, past, actions):
        # Causal sequence: actions leaving each context frame, then the current candidate.
        forecasts = []
        for h in range(actions.shape[1]):
            acts = torch.cat([past, actions[:, h : h + 1]], 1)[:, -3:]
            pred = self.model.predict(z[:, -3:], self.action_embeddings(acts))[:, -1:]
            forecasts.append(pred)
            z = torch.cat([z, pred], 1)[:, -3:]
            past = acts[:, -2:]
        return torch.cat(forecasts, 1)

    @torch.no_grad()
    def plan(self, z, past, generators):
        # z: [environments, observed keyframes, 192], past: [envs, keyframes-1,10].
        n = len(z)
        ns = 256
        h = 5
        mu = torch.zeros((n, h, self.block_dim), device="cuda")
        std = torch.ones_like(mu)
        zbase = z[:, None].expand(-1, ns, -1, -1).reshape(n * ns, z.shape[1], 192)
        ab = past[:, None].expand(-1, ns, -1, -1).reshape(n * ns, past.shape[1], self.block_dim)
        discounts = (0.99 ** torch.arange(1, h + 1, device="cuda")).view(1, h)
        with amp():
            for _ in range(6):
                noise = torch.stack(
                    [
                        torch.randn((ns, h, self.block_dim), generator=g, device="cuda")
                        for g in generators
                    ]
                )
                cand = (mu[:, None] + std[:, None] * noise).clamp_(-1, 1)
                pred = self.rollout(zbase, ab, cand.reshape(n * ns, h, self.block_dim))
                value = (
                    (self.head(pred).squeeze(-1).float() * discounts).sum(1).view(n, ns)
                )
                best = value.topk(32, 1).indices
                elite = cand[torch.arange(n, device="cuda")[:, None], best]
                mu = elite.mean(1)
                std = elite.std(1, unbiased=False).clamp_(min=0.05)
        return mu[:, 0].view(n, 5, self.action_dim).cpu().numpy()

    @torch.no_grad()
    def frames(self, envs):
        x = np.stack([e.render().transpose(2, 0, 1) for e in envs])
        with amp():
            return encode(self.model, torch.from_numpy(x).cuda()).float()


class Replay:
    def __init__(self, budget, planner, seed, horizon=50):
        cap = budget * (horizon - 18) + 32
        self.obs = torch.empty(cap, 4, 192, device="cuda")
        self.acts = torch.empty(cap, 3, planner.block_dim, device="cuda")
        self.count = self.updates = 0
        self.rng = np.random.default_rng(seed)
        self.planner = planner
        mods = (
            planner.model.action_encoder,
            planner.model.predictor,
            planner.model.pred_proj,
        )
        self.pars = [p for m in mods for p in m.parameters()]
        for p in self.pars:
            p.requires_grad_(True)
        self.opt = torch.optim.AdamW(self.pars, lr=5e-5, weight_decay=1e-3)
        self.losses = []

    def extend(self, observations, actions, first):
        # C+1=4 keyframes, K=5, only clips with all 20 raw observation rows.
        stop = max(0, len(observations) - 19)
        if first >= stop:
            return first
        z = torch.stack(observations)
        a = torch.from_numpy(np.stack(actions)).to("cuda")
        windows = torch.arange(first, stop, device="cuda")
        end = self.count + len(windows)
        self.obs[self.count : end] = z[
            windows[:, None] + torch.arange(4, device="cuda") * 5
        ]
        self.acts[self.count : end] = a[
            windows[:, None] + torch.arange(15, device="cuda")
        ].view(-1, 3, self.planner.block_dim)
        self.count = end
        if self.updates < self.count // 64:
            ix = torch.from_numpy(self.rng.choice(self.count, 64, replace=False))
            z = self.obs[ix]
            a = self.acts[ix]
            with amp():
                pred = self.planner.model.predict(
                    z[:, :3], self.planner.action_embeddings(a)
                )
            loss = (pred.float() - z[:, 1:]).square().mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pars, 1.0)
            self.opt.step()
            self.updates += 1
            self.losses.append(loss.item())
        return stop


@torch.no_grad()
def first_frame(planner, envs):
    return planner.frames(envs)


def episodes(
    planner, seeds, shift, *, replay=None, save_traj=False, fixed_horizon=False
):
    mapping = {False: None, "grid": grid_action, "wave": wave_action}[shift]
    envs = [make_eval_env(s) for s in seeds]
    n = len(seeds)
    if replay is not None and n != 1:
        raise ValueError("Online deployment is sequential")
    generators = [
        torch.Generator(device="cuda").manual_seed(int(s + 3000000)) for s in seeds
    ]
    z = first_frame(planner, envs)
    history = deque([z], maxlen=3)
    past = deque(maxlen=2)
    all_z = [z[0]] if replay is not None else []
    all_actions = []
    next_start = 0
    ended = np.zeros(n, bool)
    success = np.zeros(n, bool)
    end_score = np.zeros(n)
    peak_score = np.zeros(n)
    steps = np.zeros(n, int)
    traces_z = [z] if save_traj else []
    traces_a = []
    for k in range(10):
        ac = torch.stack(list(past), 1) if past else z.new_empty((n, 0, 10))
        block = planner.plan(torch.stack(list(history), 1), ac, generators)
        for t in range(5):
            for i, e in enumerate(envs):
                if ended[i]:
                    continue
                cmd = block[i, t]
                applied = mapping(cmd, e.agent_position.numpy()) if mapping else cmd
                _, _, done, _, info = e.step(applied)
                cur = float(score(info["state"]))
                end_score[i] = cur
                peak_score[i] = max(peak_score[i], cur)
                steps[i] += 1
                success[i] |= bool(done)
                if done and not fixed_horizon:
                    ended[i] = True
            if replay is not None or save_traj:
                z = planner.frames(envs)
                if replay is not None:
                    all_z.append(z[0])
                    all_actions.append(block[0, t].copy())
                if save_traj:
                    traces_z.append(z)
                    traces_a.append(block[:, t].copy())
            if ended.all():
                break
        if replay is not None:
            next_start = replay.extend(all_z, all_actions, next_start)
        if ended.all():
            break
        if replay is None and not save_traj:
            z = planner.frames(envs)
        history.append(z)
        past.append(torch.from_numpy(block.reshape(n, 10)).cuda())
    result = {
        "seeds": list(map(int, seeds)),
        "end_score": end_score.tolist(),
        "peak_score": peak_score.tolist(),
        "success": success.astype(int).tolist(),
        "steps": steps.tolist(),
    }
    if save_traj:
        result["traj_z"] = torch.stack(traces_z, 1).cpu()
        result["traj_a"] = torch.from_numpy(np.stack(traces_a, 1))
    return result


def evaluate(planner, seeds, shift, chunk=4):
    logs = []
    for start in range(0, len(seeds), chunk):
        logs.append(episodes(planner, seeds[start : start + chunk], shift))
    out = {k: sum([d[k] for d in logs], []) for k in logs[0]}
    return {
        "mean_end": float(np.mean(out["end_score"])),
        "mean_peak": float(np.mean(out["peak_score"])),
        "success_rate": float(np.mean(out["success"])),
        "episodes": out,
    }


def diagnostic(planner, data):
    z = data["z"].cuda()
    a = data["a"].cuda()
    ix = torch.arange(z.shape[1] - 35, device="cuda")
    latent = z[:, ix[:, None] + torch.arange(8, device="cuda") * 5].reshape(-1, 8, 192)
    acts = a[:, ix[:, None] + torch.arange(35, device="cuda")].reshape(-1, 7, planner.block_dim)
    all_mse = []
    with torch.no_grad(), amp():
        for i in range(0, len(latent), 256):
            zs = latent[i : i + 256]
            ac = acts[i : i + 256]
            pr = planner.rollout(zs[:, :3], ac[:, :2], ac[:, 2:])
            all_mse.append((pr.float() - zs[:, 3:]).square().mean(-1).cpu())
    ms = torch.cat(all_mse).view(len(z), len(ix), 5).mean(1).mean(0)
    return {
        "five_block_mse": ms.mean().item(),
        "by_horizon": ms.tolist(),
        "trajectories": len(z),
    }


def get_planner(name):
    m = model_init()
    weights(m, ROOT / f"models/{name}-weights.pt")
    m.cuda().eval().requires_grad_(False)
    ck = torch.load(
        ROOT / f"models/{name}-reward.pt", map_location="cpu", weights_only=True
    )
    head = head_init()
    head.load_state_dict(ck["head"])
    head.cuda().eval().requires_grad_(False)
    return Planner(m, head, ck["mean"], ck["std"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["frozen", "ttt"])
    ap.add_argument("--base", choices=["scratch"], default="scratch")
    ap.add_argument("--shift", choices=["grid", "wave"], default="grid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--period", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--diag-n", type=int, default=100)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    path = ROOT / "runs" / args.out
    path.mkdir(parents=True, exist_ok=True)
    planner = get_planner(args.base)
    seeds = list(range(1000000, 1000000 + args.eval_n))
    print("start", vars(args), time.ctime(), flush=True)
    meta = {
        "config": vars(args),
        "score_convention": "terminal task score; also record peak and strict success",
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "initial_predictor_sha": digest(
            planner.model, ["predictor", "pred_proj", "action_encoder"]
        ),
        "encoder_sha": digest(planner.model, ["encoder", "projector"]),
    }
    (ROOT / "data").mkdir(exist_ok=True)
    diagpath = ROOT / "data" / f"{args.base}-{args.shift}-frozen-traj-{args.diag_n}.pt"
    if args.mode == "frozen":
        for shift in [False, args.shift]:
            res = evaluate(planner, seeds, shift, args.chunk)
            lbl = args.shift if shift else "original"
            meta[lbl] = res
            write_json(path / "frozen.json", meta)
            print(
                lbl,
                {k: v for k, v in res.items() if k != "episodes"},
                time.ctime(),
                flush=True,
            )
        if args.diag_n:
            zs = []
            aa = []
            for i in range(0, args.diag_n, args.chunk):
                r = episodes(
                    planner,
                    list(
                        range(2000000 + i, 2000000 + min(i + args.chunk, args.diag_n))
                    ),
                    args.shift,
                    save_traj=True,
                    fixed_horizon=True,
                )
                zs.append(r["traj_z"])
                aa.append(r["traj_a"])
            data = {"z": torch.cat(zs), "a": torch.cat(aa)}
            torch.save(data, diagpath)
            meta.update(diagnostic(planner, data))
            write_json(path / "frozen.json", meta)
        print("finished frozen", time.ctime(), flush=True)
    else:
        baseline = json.loads((path.parent / "frozen" / "frozen.json").read_text())
        assert (
            baseline["config"].get("shift", "grid") == args.shift
            and baseline["config"]["base"] == args.base
            and baseline[args.shift]["episodes"]["seeds"] == seeds
        )
        assert meta["initial_predictor_sha"] == baseline["initial_predictor_sha"]
        replay = Replay(args.episodes, planner, 110000 + args.run)
        meta["checkpoints"] = [
            {"episode": 0, "nwindows": 0, "updates": 0, "eval": baseline[args.shift]}
        ]
        records = []
        for episode in range(1, args.episodes + 1):
            r = episodes(
                planner,
                [4000000 + 100000 * args.run + episode],
                args.shift,
                replay=replay,
                fixed_horizon=True,
            )
            records.append(r["end_score"][0])
            if episode % 10 == 0:
                print(
                    "online",
                    episode,
                    "windows",
                    replay.count,
                    "updates",
                    replay.updates,
                    "score_last10",
                    np.mean(records[-10:]),
                    time.ctime(),
                    flush=True,
                )
            if episode % args.period == 0 or episode == args.episodes:
                ev = evaluate(planner, seeds, args.shift, args.chunk)
                meta["checkpoints"].append(
                    {
                        "episode": episode,
                        "nwindows": replay.count,
                        "updates": replay.updates,
                        "eval": ev,
                    }
                )
                meta["online_scores"] = records
                meta["loss"] = replay.losses
                meta["last_predictor_sha"] = digest(
                    planner.model, ["predictor", "pred_proj", "action_encoder"]
                )
                meta["end_encoder_sha"] = digest(
                    planner.model, ["encoder", "projector"]
                )
                assert (
                    meta["encoder_sha"] == meta["end_encoder_sha"]
                    and replay.updates == replay.count // 64
                )
                if args.diag_n:
                    meta["checkpoints"][-1].update(
                        diagnostic(planner, torch.load(diagpath, weights_only=True))
                    )
                write_json(path / "ttt.json", meta)
                torch.save(
                    {
                        k: v
                        for k, v in planner.model.state_dict().items()
                        if k.split(".")[0]
                        in ("action_encoder", "pred_proj", "predictor")
                    },
                    path / "predictor.pt",
                )
                print(
                    "heldout",
                    episode,
                    ev["mean_end"],
                    ev["mean_peak"],
                    ev["success_rate"],
                    flush=True,
                )
