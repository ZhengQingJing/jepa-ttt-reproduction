"""Goal-image-free episodic AdaJEPA comparison on the two JEPA-TTT Two-Room shifts."""

import argparse
from collections import deque
import statistics
import numpy as np
import torch
from common import ROOT, amp, grid_action, wave_action, make_eval_env, normalize, score
from experiment import get_planner, write_json


@torch.no_grad()
def observe(planner, env):
    frame = torch.as_tensor(
        env.render().transpose(2, 0, 1).copy(), device="cuda"
    ).unsqueeze(0)
    with amp():
        return planner.model.encoder(normalize(frame)).last_hidden_state[:, 0].float()


def adapt(planner, features, blocks, params):
    model = planner.model
    # The model uses C=3 and K=5. Merge the most recent complete action segments.
    T = len(blocks)
    width = min(3, T)
    with amp():
        z = model.projector(torch.cat(list(features), 0)).unsqueeze(0)
        a = torch.as_tensor(np.asarray(blocks, np.float32), device="cuda").reshape(
            1, T, 10
        )
        losses = []
        for t in range(T - width + 1):
            pred = model.predict(
                z[:, t : t + width], planner.action_embeddings(a[:, t : t + width])
            )
            losses.append(
                (pred.float() - z[:, t + 1 : t + 1 + width].detach().float())
                .square()
                .mean()
            )
        loss = torch.stack(losses).mean()
    opt = torch.optim.Adam(params)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.detach())


def trial(planner, seed, transform, params):
    env = make_eval_env(seed)
    features = deque([observe(planner, env)], maxlen=6)
    commands = deque(maxlen=5)
    gen = torch.Generator(device="cuda").manual_seed(seed + 3000000)
    updates = 0
    peak = 0.0
    for k in range(10):
        with torch.no_grad(), amp():
            z = (
                planner.model.projector(torch.cat(list(features)[-3:], 0))
                .float()
                .unsqueeze(0)
            )
        hist = list(commands)[-2:]
        past = (
            torch.as_tensor(np.stack(hist), device="cuda").reshape(1, len(hist), 10)
            if hist
            else z.new_empty((1, 0, 10))
        )
        block = planner.plan(z, past, [gen])[0]
        for j, cmd in enumerate(block, 1):
            _, _, done, _, info = env.step(transform(cmd, env.agent_position.numpy()))
            end = float(score(info["state"]))
            peak = max(peak, end)
            if done:
                return {
                    "seed": seed,
                    "steps": 5 * k + j,
                    "end_score": end,
                    "peak_score": peak,
                    "success": 1,
                    "updates": updates,
                }
        commands.append(block.copy())
        features.append(observe(planner, env))
        adapt(planner, features, commands, params)
        updates += 1
    return {
        "seed": seed,
        "steps": 50,
        "end_score": end,
        "peak_score": peak,
        "success": 0,
        "updates": updates,
    }


def run(args):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    planner = get_planner("scratch")
    m = planner.model
    mods = [m.predictor.transformer.layers[-1], m.predictor.transformer.norm]
    groups = [{"params": [p for mod in mods for p in mod.parameters()], "lr": 5e-4}]
    if args.encoder == "trainable":
        mods.append(m.projector)
        groups.append({"params": list(m.projector.parameters()), "lr": 1e-5})
    for mod in mods:
        mod.requires_grad_(True)
    initial = [
        {k: v.detach().clone() for k, v in mod.state_dict().items()} for mod in mods
    ]
    out = (
        ROOT
        / "runs"
        / "episodic-baseline"
        / args.shift
        / args.encoder
        / f"run{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for i in range(100):
        for mod, state in zip(mods, initial):
            mod.load_state_dict(state)
        row = trial(
            planner,
            1000000 + 10000 * args.seed + i,
            {"grid": grid_action, "wave": wave_action}[args.shift],
            groups,
        )
        rows.append(row)
        if len(rows) % 10 == 0:
            mean = statistics.mean(x["end_score"] for x in rows)
            write_json(
                out,
                {
                    "shift": args.shift,
                    "encoder_head": args.encoder,
                    "seed": args.seed,
                    "mean": mean,
                    "episodes": rows,
                },
            )
            print(len(rows), mean, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shift", choices=["grid", "wave"], required=True)
    p.add_argument("--encoder", choices=["frozen", "trainable"], required=True)
    p.add_argument("--seed", type=int, choices=range(3), required=True)
    run(p.parse_args())
