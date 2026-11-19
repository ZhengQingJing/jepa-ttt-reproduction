"""Matched cross-episode persistence control (JEPA-TTT, Appendix A6.1)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import ROOT, amp
from experiment import digest, episodes, evaluate, get_planner, write_json


MODULES = ("action_encoder", "predictor", "pred_proj")
BATCH = 8
CONTEXT = 3
FRAME_SKIP = 5


class EpisodeLocalUpdater:
    """Stream the current episode's dense windows once, in chronological batches."""

    def __init__(self, planner):
        self.planner = planner
        self.parameters = [
            param for name in MODULES
            for param in getattr(planner.model, name).parameters()
        ]
        for param in self.parameters:
            param.requires_grad_(True)
        self.initial = {
            name: {key: tensor.detach().clone() for key, tensor in
                   getattr(planner.model, name).state_dict().items()}
            for name in MODULES
        }
        self.optimizer = self.new_optimizer()
        self.updates = 0
        self.begin_episode(False)

    def new_optimizer(self):
        return torch.optim.AdamW(self.parameters, lr=5e-5, weight_decay=1e-3)

    def begin_episode(self, reset):
        self.trained_windows = 0
        if reset:
            for name in MODULES:
                getattr(self.planner.model, name).load_state_dict(self.initial[name])
            self.optimizer = self.new_optimizer()

    def extend(self, observations, actions, first):
        # A start becomes eligible once (C + 1) * K observation rows exist.
        stop = max(0, len(observations) - (CONTEXT + 1) * FRAME_SKIP + 1)
        if stop - self.trained_windows < BATCH:
            return stop
        origins = torch.arange(self.trained_windows, self.trained_windows+BATCH,
                               device=observations[0].device)
        offsets = torch.arange(CONTEXT+1, device=origins.device) * FRAME_SKIP
        action_offsets = torch.arange(CONTEXT * FRAME_SKIP, device=origins.device)
        z = torch.stack(observations)[origins[:,None]+offsets]
        a = torch.as_tensor(np.stack(actions), device=origins.device)
        a = a[origins[:,None]+action_offsets].reshape(BATCH, CONTEXT, -1)
        with amp():
            pred = self.planner.model.predict(z[:, :-1], self.planner.action_embeddings(a))
        loss = (pred.float()-z[:, 1:]).square().mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
        self.optimizer.step()
        self.trained_windows += BATCH
        self.updates += 1
        return stop


def run(args):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    planner = get_planner('scratch')
    original_encoder = digest(planner.model, {'encoder', 'projector'})
    updater = EpisodeLocalUpdater(planner)
    evaluation_seeds = list(range(1000000, 1000100))
    output = args.output / args.shift / args.arm / f'run{args.run}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    def measure(episode):
        with torch.no_grad():
            value = evaluate(planner, evaluation_seeds, args.shift, chunk=4)
        return {'episode': episode, 'updates': updater.updates,
                'mean_end': value['mean_end'], 'mean_peak': value['mean_peak'],
                'success_rate': value['success_rate']}

    records = [measure(0)]
    for ep in range(1, 501):
        updater.begin_episode(args.arm == 'reset')
        episode_seed = 4000000 + 100000*args.run + ep
        episodes(planner, [episode_seed], args.shift, replay=updater, fixed_horizon=True)
        assert updater.trained_windows == 32 and updater.updates == 4*ep
        if ep % 50 == 0:
            record = measure(ep)
            records.append(record)
            assert digest(planner.model, {'encoder','projector'}) == original_encoder
            write_json(output, {'shift': args.shift, 'arm': args.arm, 'run': args.run,
                                'batch': BATCH, 'eval_seed_start': evaluation_seeds[0],
                                'eval_size': len(evaluation_seeds), 'checkpoints': records})
            print(args.shift, args.arm, args.run, record, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shift', choices=('grid', 'wave'), required=True)
    parser.add_argument('--arm', choices=('persistent', 'reset'), required=True)
    parser.add_argument('--run', type=int, choices=(0,1,2), required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'runs'/'persistence')
    run(parser.parse_args())
