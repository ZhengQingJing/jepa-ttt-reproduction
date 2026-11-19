"""OGBench-Cube interface for the common JEPA-TTT planning/deployment driver."""

import atexit
import os
from pathlib import Path
import torch
from cube_env import Hover
from cube_model import Planner, head_init, make_model


def load_planner():
    directory = Path(os.environ.get("CUBE_MODELS", "models"))
    model = make_model()
    model.load_state_dict(
        torch.load(directory / "cube-weights.pt", map_location="cpu", weights_only=True)
    )
    model.cuda().eval().requires_grad_(False)
    state = torch.load(
        directory / "cube-reward.pt", map_location="cpu", weights_only=True
    )
    head = head_init()
    head.load_state_dict(state["head"])
    head.cuda().eval().requires_grad_(False)
    planner = Planner(model, head, state["mean"], state["std"])
    assert planner.action_dim == 3 and planner.block_dim == 15
    return planner


_available = []
_close_registered = False


def close_renderers():
    for env in _available:
        env.close()


class Environment:
    def __init__(self, simulator):
        self.simulator = simulator

    def render(self):
        global _close_registered
        frame = self.simulator.render()
        if not _close_registered:
            # Register after MuJoCo opens EGL so its teardown runs after ours.
            atexit.register(close_renderers)
            _close_registered = True
        return frame

    def step(self, command):
        _, value, reached = self.simulator.step(command)
        return value, reached

    def close(self):
        if self.simulator is not None:
            _available.append(self.simulator)
            self.simulator = None


def make_env(seed, shift):
    # A native reset reinitializes MjData, robot, cube, RNG, and the delayed command.
    # Reusing compiled models avoids re-parsing meshes at each held-out episode.
    env = _available.pop() if _available else Hover(shift)
    env.shift = shift
    env.reset(seed)
    return Environment(env)
