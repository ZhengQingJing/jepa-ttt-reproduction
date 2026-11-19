"""Frozen planning and persistent JEPA-TTT under the paper's task-specific shifts."""

import argparse
from collections import deque
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

from experiment import Replay, diagnostic, digest, write_json


def load_adapter(path):
    path = path.resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("benchmark_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def interact(
    adapter, planner, seeds, shift, horizon, *, replay=None, record=False, peak=False
):
    n = len(seeds)
    if replay is not None and n != 1:
        raise ValueError("Each online deployment is sequential.")
    full_horizon = replay is not None or record
    with ExitStack() as resources:
        envs = []
        for seed in seeds:
            env = adapter.make_env(seed, shift)
            resources.callback(env.close)
            envs.append(env)
        generators = [
            torch.Generator(device="cuda").manual_seed(s + 3000000) for s in seeds
        ]
        behavior = (
            [adapter.make_behavior(s) for s in seeds]
            if record and hasattr(adapter, "make_behavior")
            else None
        )
        z = planner.frames(envs)
        history, past = deque([z], maxlen=3), deque(maxlen=2)
        embeddings = [z] if full_horizon else []
        commands = []
        first = 0
        ended = np.zeros(n, bool)
        successes = np.zeros(n, bool)
        terminal = np.zeros(n)
        peaks = np.zeros(n)
        lengths = np.zeros(n, int)
        for start in range(0, horizon, 5):
            if behavior is None:
                previous = (
                    torch.stack(list(past), 1)
                    if past
                    else z.new_empty((n, 0, planner.block_dim))
                )
                block = planner.plan(
                    torch.stack(list(history), 1), previous, generators
                )
            else:
                block = np.empty((n, 5, planner.action_dim), dtype=np.float32)
            for step in range(5):
                for index, env in enumerate(envs):
                    if ended[index]:
                        continue
                    if behavior is not None:
                        block[index, step] = behavior[index](env, start + step)
                    value, reached = env.step(block[index, step])
                    lengths[index] += 1
                    terminal[index] = value
                    peaks[index] = max(peaks[index], value)
                    successes[index] |= reached
                    ended[index] = reached and not full_horizon
                if full_horizon:
                    z = planner.frames(envs)
                    embeddings.append(z)
                    commands.append(block[:, step].copy())
                if ended.all():
                    break
            if replay is not None:
                first = replay.extend(
                    [x[0] for x in embeddings], [x[0] for x in commands], first
                )
            if ended.all():
                break
            if not full_horizon:
                z = planner.frames(envs)
            history.append(z)
            past.append(torch.from_numpy(block.reshape(n, planner.block_dim)).cuda())
        logs = [
            {
                "seed": s,
                "score": float((peaks if peak else terminal)[i]),
                "terminal_score": float(terminal[i]),
                "peak_score": float(peaks[i]),
                "steps": int(lengths[i]),
                "success": bool(successes[i]),
            }
            for i, s in enumerate(seeds)
        ]
        if record:
            return torch.stack(embeddings, dim=1).cpu(), torch.from_numpy(
                np.stack(commands, axis=1)
            )
        return logs


def evaluate(adapter, planner, shift, horizon, peak, size, chunk):
    rows = []
    for i in range(0, size, chunk):
        seeds = list(range(1000000 + i, 1000000 + min(i + chunk, size)))
        rows.extend(interact(adapter, planner, seeds, shift, horizon, peak=peak))
    return {
        "mean": float(np.mean([r["score"] for r in rows])),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "episodes": rows,
    }


def run(args):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    adapter = load_adapter(args.adapter)
    planner = adapter.load_planner()
    planner.model.eval().requires_grad_(False)
    planner.head.eval().requires_grad_(False)
    horizon = 300 if args.task == "pusht" else 50
    peak = args.task == "pusht"
    base = args.output.resolve() / args.task / args.shift
    base.mkdir(parents=True, exist_ok=True)
    frozen_file = base / "frozen.json"
    diagnostic_file = base / "diagnostic.pt"
    trainable = {"action_encoder", "predictor", "pred_proj"}
    hash0 = digest(planner.model, trainable)
    enc = digest(planner.model, {"encoder", "projector"})
    head_hash = digest(
        planner.head, {key.split(".")[0] for key in planner.head.state_dict()}
    )
    settings = {
        "task": args.task,
        "shift": args.shift,
        "horizon": horizon,
        "eval_n": args.eval_n,
        "chunk": args.chunk,
        "initial_predictor_sha": hash0,
        "encoder_sha": enc,
        "reward_sha": head_hash,
    }
    if args.mode == "frozen":
        if frozen_file.exists():
            raise FileExistsError(frozen_file)
        logs = {
            key: evaluate(adapter, planner, key, horizon, peak, args.eval_n, args.chunk)
            for key in ("original", args.shift)
        }
        states, actions = [], []
        for i in range(0, args.eval_n, args.chunk):
            seeds = list(range(2000000 + i, 2000000 + min(i + args.chunk, args.eval_n)))
            z, a = interact(
                adapter, planner, seeds, args.shift, horizon, record=True, peak=peak
            )
            states.append(z)
            actions.append(a)
        trace = dict(z=torch.cat(states), a=torch.cat(actions))
        torch.save(trace, diagnostic_file)
        write_json(
            frozen_file,
            {"settings": settings, "evaluations": logs, **diagnostic(planner, trace)},
        )
        return

    import json

    frozen = json.loads(frozen_file.read_text())
    if frozen["settings"] != settings:
        raise ValueError(
            "Frozen evaluation does not match the requested model and protocol."
        )
    trace = torch.load(diagnostic_file, map_location="cpu", weights_only=True)
    replay = Replay(args.episodes, planner, 110000 + args.seed, horizon=horizon)
    folder = base / f"run{args.seed}"
    folder.mkdir(exist_ok=True)
    final_path = folder / "ttt.json"
    if final_path.exists():
        raise FileExistsError(final_path)
    checkpoints = [
        {
            "episode": 0,
            "nwindows": 0,
            "updates": 0,
            "eval": frozen["evaluations"][args.shift],
        }
    ]
    for ep in range(1, args.episodes + 1):
        interact(
            adapter,
            planner,
            [4000000 + 100000 * args.seed + ep],
            args.shift,
            horizon,
            replay=replay,
            peak=peak,
        )
        if ep % 50 == 0 or ep == args.episodes:
            assert (
                replay.count == ep * (horizon - 18)
                and replay.updates == replay.count // 64
            )
            value = evaluate(
                adapter, planner, args.shift, horizon, peak, args.eval_n, args.chunk
            )
            checkpoints.append(
                {
                    "episode": ep,
                    "nwindows": replay.count,
                    "updates": replay.updates,
                    "eval": value,
                    **diagnostic(planner, trace),
                }
            )
            assert digest(planner.model, {"encoder", "projector"}) == enc
            result = {
                "settings": settings,
                "run": args.seed,
                "checkpoints": checkpoints,
                "final_predictor_sha": digest(planner.model, trainable),
            }
            write_json(
                final_path if ep == args.episodes else folder / "progress.json", result
            )
            print(
                args.task,
                args.shift,
                args.seed,
                ep,
                value["mean"],
                checkpoints[-1]["five_block_mse"],
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("frozen", "ttt"))
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--task", choices=("pusht", "reacher", "cube"), required=True)
    p.add_argument("--shift", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--eval-n", type=int, default=100)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--seed", type=int, choices=range(3), default=0)
    run(p.parse_args())
