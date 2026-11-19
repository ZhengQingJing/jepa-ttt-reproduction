"""Frozen planner, native environment and diagnostic behavior policy for PushT."""

from pathlib import Path

import numpy as np
import torch

if __package__:
    from .task import PushTTask
else:
    from task import PushTTask
from common import head_init, model_init
from experiment import Planner


class ScaledScore(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, z):
        return 10.0 * self.head(z)


def load_planner(models_dir=None):
    folder = (
        Path(models_dir)
        if models_dir is not None
        else Path(__file__).resolve().parent / "models"
    )
    model = model_init().cuda().eval().requires_grad_(False)
    model.load_state_dict(
        torch.load(folder / "world.pt", weights_only=True, map_location="cpu")
    )
    record = torch.load(folder / "reward.pt", weights_only=True, map_location="cpu")
    head = head_init().cuda().eval().requires_grad_(False)
    head.load_state_dict(record["head"])
    return Planner(model, ScaledScore(head).eval(), record["mean"], record["std"])


def make_env(seed, shift="original"):
    env = PushTTask(shift)
    env.reset(seed)
    return env


def make_behavior(seed):
    rng = np.random.default_rng(int(seed))
    target = None

    def sample(env, t):
        nonlocal target
        if t % 4 == 0:
            target = env.sample_command(rng)
        return target.copy()

    return sample
