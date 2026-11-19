"""Table 1 streaming controls for Two-Room; same model, planner and paired seeds."""

import argparse
from collections import deque
import numpy as np
import torch
from common import ROOT, amp
from experiment import digest, episodes, evaluate, get_planner, write_json

MODES = ("sparse", "matched-sparse", "dense")
MODULES = ("action_encoder", "predictor", "pred_proj")


class Stream:
    def __init__(self, planner, mode):
        self.planner, self.mode = planner, mode
        self.pending = deque()
        self.presentations = self.updates = 0
        self.params = [
            p for name in MODULES for p in getattr(planner.model, name).parameters()
        ]
        for p in self.params:
            p.requires_grad_(True)
        self.opt = torch.optim.AdamW(self.params, lr=5e-5, weight_decay=1e-3)

    def extend(self, obs, actions, first):
        dense_stop = max(0, len(obs) - 19)
        if self.mode == "dense":
            origins = range(first, dense_stop)
        else:
            start = len(actions) - 15
            count = 1 if self.mode == "sparse" else dense_stop - first
            origins = [start] * count if start >= 0 else []
        for t in origins:
            states = torch.stack(obs[t : t + 16 : 5])
            acts = torch.as_tensor(
                np.stack(actions[t : t + 15]), device=states.device
            ).reshape(3, 10)
            self.pending.append((states, acts))
            self.presentations += 1
        if len(self.pending) >= 64:
            inputs, commands = zip(*(self.pending.popleft() for _ in range(64)))
            z, a = torch.stack(inputs), torch.stack(commands)
            with amp():
                pred = self.planner.model.predict(
                    z[:, :3], self.planner.action_embeddings(a)
                )
            loss = (pred.float() - z[:, 1:]).square().mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
            self.updates += 1
        return dense_stop


def run(args):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    planner = get_planner("scratch")
    updater = Stream(planner, args.mode)
    eval_seeds = list(range(1000000, 1000100))
    out = (
        ROOT
        / "runs"
        / "stream-controls"
        / args.shift
        / args.mode
        / f"run{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    initial = digest(planner.model, {"encoder", "projector"})
    records = []
    for ep in range(501):
        if ep:
            episodes(
                planner,
                [4000000 + 100000 * args.seed + ep],
                args.shift,
                replay=updater,
                fixed_horizon=True,
            )
        if ep % 50 == 0:
            ev = evaluate(planner, eval_seeds, args.shift, chunk=4)
            count = ep * (8 if args.mode == "sparse" else 32)
            assert updater.presentations == count and updater.updates == count // 64
            assert digest(planner.model, {"encoder", "projector"}) == initial
            records.append(
                {
                    "episode": ep,
                    "presentations": count,
                    "updates": updater.updates,
                    "eval": ev,
                }
            )
            write_json(
                out,
                {
                    "shift": args.shift,
                    "mode": args.mode,
                    "seed": args.seed,
                    "checkpoints": records,
                },
            )
            print(ep, ev["mean_end"], updater.updates, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shift", choices=["wave", "grid"], required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--seed", type=int, choices=range(3), required=True)
    run(p.parse_args())
