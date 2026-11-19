"""Two-Room grid-rotation experiment. Paths are relative to this experiment directory."""

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import ViTConfig, ViTModel

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "upstream/stable-worldmodel"), str(ROOT / "upstream/le-wm")]
from stable_worldmodel.envs.two_room import TwoRoomEnv, ExpertPolicy as ExpertPolicy
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg as SIGReg

GOAL = np.array([164.0, 112.0], dtype=np.float32)
DOOR = np.array([112.0, 49.0], dtype=np.float32)
K, C = 5, 3


def score(pos):
    p = np.asarray(pos)
    d = np.where(
        p[..., 0] < 112,
        np.linalg.norm(p - DOOR, axis=-1) + np.linalg.norm(DOOR - GOAL),
        np.linalg.norm(p - GOAL, axis=-1),
    )
    return np.exp(-d / 64.0)


def grid_action(action, pos):
    x, y = pos
    angle = (
        math.radians(75)
        * math.sin(2 * math.pi * (x - 20) / 96)
        * math.cos(2 * math.pi * (y - 20) / 96)
    )
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(
        [c * action[0] - s * action[1], s * action[0] + c * action[1]], dtype=np.float32
    )


def wave_action(action, pos):
    angle = math.radians(75) * math.sin(2 * math.pi * (pos[0] - 20) / 88)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(
        [c * action[0] - s * action[1], s * action[0] + c * action[1]], dtype=np.float32
    )


def model_init():
    cfg = ViTConfig(
        image_size=224,
        patch_size=14,
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
    )
    cfg._attn_implementation = "sdpa"
    encoder = ViTModel(cfg, add_pooling_layer=False, use_mask_token=False)
    pred = ARPredictor(
        num_frames=3,
        input_dim=192,
        hidden_dim=192,
        output_dim=192,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )
    return JEPA(
        encoder,
        pred,
        Embedder(input_dim=10, emb_dim=192),
        MLP(192, 2048, 192, nn.BatchNorm1d),
        MLP(192, 2048, 192, nn.BatchNorm1d),
    )


def head_init():
    return nn.Sequential(
        nn.Linear(192, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )


def normalize(x):
    mean = x.new_tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, :, None, None]
    return (x.float() / 255.0 - mean) / std


def encode(model, pixels):
    return model.projector(model.encoder(normalize(pixels)).last_hidden_state[:, 0])


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16)


class OfflineRenderer:
    """Regenerate unaugmented upstream RGB frames from the compact offline trajectories."""

    def __init__(self, env):
        wall, door = env._wall_and_door_masks()
        img = torch.full((3, 224, 224), 255, dtype=torch.float32)
        img[:, door] = 255
        img[:, wall] = 0
        self.image = img.cuda()
        self.x, self.y = env.grid_x.cuda(), env.grid_y.cuda()
        self.color = torch.tensor([255.0, 0.0, 0.0], device="cuda").view(1, 3, 1, 1)
        self.radius = float(env.variation_space["agent"]["radius"].value.item())

    def __call__(self, poses):
        px, py = poses[:, 0, None, None], poses[:, 1, None, None]
        d2 = (self.x - px).square() + (self.y - py).square()
        alpha = torch.exp(-d2 / (2.0 * self.radius**2))
        alpha = (alpha / alpha.amax((-2, -1), keepdim=True)).clamp(0, 1)[:, None]
        return (self.image[None] * (1.0 - alpha) + self.color * alpha).to(torch.uint8)


def make_eval_env(seed):
    # The released environment samples starts in both rooms. Condition its reset on the left room.
    env = TwoRoomEnv()
    for attempt in range(512):
        _, info = env.reset(
            seed=int(seed + attempt * 100003), options={"target_state": GOAL}
        )
        if info["state"][0] < 100.0:
            return env
    raise RuntimeError("Cannot draw a valid start from the left room")
