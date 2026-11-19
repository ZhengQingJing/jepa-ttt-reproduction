"""The published LeWM backbone with 15 inputs for a block of five XYZ commands."""

import os
import sys
from pathlib import Path

import torch

repo = Path(
    os.environ.get("JEPA_TTT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
sys.path.insert(0, str(repo / "src"))
from common import Embedder, SIGReg, amp, encode, head_init, model_init
from experiment import Planner


def make_model():
    model = model_init()
    model.action_encoder = Embedder(input_dim=15, emb_dim=192)
    return model
