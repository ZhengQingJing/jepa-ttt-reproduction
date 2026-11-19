"""Reacher adapter for the shared dense-replay deployment driver."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reacher_task import MODELS, Reacher, score, success
from common import head_init, model_init
from experiment import Planner


def load_planner():
    model = model_init().cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(MODELS/"weights.pt",map_location="cpu",weights_only=True))
    checkpoint = torch.load(MODELS/"reward.pt",map_location="cpu",weights_only=True)
    head = head_init().cuda().eval().requires_grad_(False)
    head.load_state_dict(checkpoint["head"])
    return Planner(model, head, checkpoint["mean"], checkpoint["std"])


class DeploymentEnv:
    def __init__(self, seed, shift):
        self.native = Reacher()
        self.native.reset(seed)
        self.shift = shift

    def render(self):
        return self.native.render()

    def step(self, command):
        q = self.native.step(command, self.shift)
        return float(score(q)), success(q)

    def close(self):
        self.native.close()


def make_env(seed, shift):
    if shift not in ("original","joint-phase","harmonic"):
        raise ValueError(shift)
    return DeploymentEnv(seed, shift)
