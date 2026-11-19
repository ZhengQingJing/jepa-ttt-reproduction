"""Reacher joint-space task and test-time fields (JEPA-TTT, Appendix A2.3)."""
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upstream/stable-worldmodel"))
os.environ.setdefault("MUJOCO_GL", "egl")
from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

DATA = ROOT / "data/reacher"
MODELS = ROOT / "models/reacher"
RUNS = ROOT / "runs/reacher"
EPISODES = 3000
TRAIN_STEPS = 200
DEPLOY_STEPS = 50


def wrapped(q):
    return (np.asarray(q) + np.pi) % (2 * np.pi) - np.pi


def score(q):
    return np.exp(-np.mean(wrapped(q) ** 2, axis=-1) / 2)


def success(q):
    return bool(np.all(np.abs(wrapped(q)) < 0.05))


def applied(action, q, shift):
    if shift == "original":
        return np.asarray(action).copy()
    q = wrapped(q)
    if shift == "joint-phase":
        psi = q[0] + 0.5 * q[1]
        max_angle, kappa = 155.0, 0.35
    elif shift == "harmonic":
        psi = math.sin(q[0]) + 0.7 * math.sin(q[1])
        max_angle, kappa = 165.0, 0.30
    else:
        raise ValueError(shift)
    gate = -math.expm1(-float(np.mean(q**2)) / (2 * 0.20**2))
    angle = math.radians(max_angle) * gate * math.tanh(psi / kappa)
    c, s = math.cos(angle), math.sin(angle)
    return np.clip([c * action[0] - s * action[1], s * action[0] + c * action[1]], -1, 1)


class Reacher:
    def __init__(self):
        # Keep the underlying suite episode alive after a strict goal hit; scoring
        # and held-out termination are evaluated after each raw Gym action.
        self.env = ReacherDMControlWrapper(task="qpos_match", seed=3072)

    def reset(self, seed):
        self.env.reset(seed=int(seed))

    @property
    def position(self):
        return self.env.env.physics.data.qpos.copy()

    def step(self, command, shift):
        self.env.step(applied(command, self.position, shift))
        return self.position

    def render(self):
        return self.env.render(width=224, height=224, camera_id=0)

    def close(self):
        self.env.close()
