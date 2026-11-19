"""JEPA-TTT Appendix A2.4 on the OGBench single-cube MuJoCo simulator."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from ogbench.manipspace.envs.cube_env import CubeEnv

GOAL = np.asarray([0.425, 0.0, 0.12], dtype=np.float64)
CUBE_XY = GOAL[:2]
SHIFTS = ("original", "quadratic", "delayed")
ACTION_DIM = 3
HORIZON = 50


def transformed(command, previous, shift):
    x, y, z = np.asarray(command, dtype=np.float64)
    if shift == "original":
        return np.asarray([x, y, z])
    if shift == "quadratic":
        return np.clip([x + 3 * y * z, y + 3 * z * x, z + 3 * x * y], -1.0, 1.0)
    if shift == "delayed":
        return np.asarray(previous)[[1, 2, 0]].copy()
    raise ValueError(shift)


def task_score(position):
    return np.exp(-np.linalg.norm(position - GOAL, axis=-1) / 0.10)


class Hover:
    def __init__(self, shift="original"):
        if shift not in SHIFTS:
            raise ValueError(shift)
        self.shift = shift
        self.raw = CubeEnv(
            env_type="single",
            ob_type="states",
            mode="data_collection",
            visualize_info=False,
            terminate_at_goal=False,
            width=224,
            height=224,
        )
        self.raw._object_sampling_bounds = np.stack([CUBE_XY, CUBE_XY])
        self.previous = np.zeros(ACTION_DIM, np.float64)

    def reset(self, seed):
        self.previous.fill(0.0)
        _, info = self.raw.reset(seed=int(seed))
        self.position = np.array(info["proprio/effector_pos"], copy=True)
        truth = self.raw._data.joint("object_joint_0").qpos[:3]
        np.testing.assert_allclose(truth, [*CUBE_XY, 0.02], atol=0.001)
        return self.position.copy()

    def render(self):
        return self.raw.render(camera="front")

    def step(self, command):
        command = np.asarray(command, np.float64)
        if command.shape != (3,) or np.abs(command).max() > 1.000001:
            raise ValueError("Expected a normalized xyz command.")
        self.applied = transformed(command, self.previous, self.shift)
        self.previous = command.copy()
        _, _, _, _, info = self.raw.step(np.r_[self.applied, 0.0, 0.0])
        self.position = np.array(info["proprio/effector_pos"], copy=True)
        distance = float(np.linalg.norm(self.position - GOAL))
        return self.position.copy(), float(np.exp(-distance / 0.10)), distance <= 0.05

    def close(self):
        if self.raw._renderer is not None:
            self.raw._renderer.close()
            self.raw._renderer = None
        self.raw.close()
