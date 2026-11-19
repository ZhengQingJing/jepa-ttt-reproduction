# Reacher

This task uses the pinned stable-worldmodel Reacher wrapper, DeepMind Control Suite and MuJoCo. The target is the joint configuration `(0, 0)`. For wrapped joint angles `q`, the task score is `exp(-mean(q²)/2)`; a strict goal hit requires both angles to have magnitude below 0.05 radians. The shifted command is evaluated from the two joint angles before the upstream action step. The planner receives camera-0 RGB observations and executed commands.

| Shift | Field | Saturation scale | Maximum rotation |
|---|---|---:|---:|
| `joint-phase` | q₁ + 0.5 q₂ | 0.35 | 155° |
| `harmonic` | sin(q₁) + 0.7 sin(q₂) | 0.30 | 165° |

The rotation angle is multiplied by the goal gate from Eq. (9); both actuator channels are clipped to [-1, 1] after rotation. [JEPA-TTT, Appendix A2.3](https://arxiv.org/pdf/2610.00722#page=21)

From the repository root after installing the main requirements and running `scripts/setup_upstream.py`:

```bash
python -m pip install -r experiments/reacher/requirements.txt
export MUJOCO_GL=egl
python experiments/reacher/collect.py --workers 4
python experiments/reacher/verify_reacher.py --collection --output runs/reacher/collection-check.json
python experiments/reacher/offline_reacher.py world
python experiments/reacher/offline_reacher.py reward
```

For each shifted environment, run the frozen controller first to write its 100 fixed diagnostic trajectories. Then start the three deployment streams from the same checkpoint. Each stream runs for 500 episodes and evaluates the original fixed suite every 50 episodes.

```bash
for shift in joint-phase harmonic; do
  python src/benchmark.py frozen --task reacher --shift "$shift" --adapter experiments/reacher/online_adapter.py --output runs
  for seed in 0 1 2; do
    python src/benchmark.py ttt --task reacher --shift "$shift" --adapter experiments/reacher/online_adapter.py --output runs --seed "$seed"
  done
done
```

Collection writes approximately 90.3 GB of native RGB frames, plus the aligned joint angles and commanded actions. Each of the 3,000 original-dynamics episodes contains 200 observations and 200 uniform random two-dimensional commands. Collection uses separate deterministic environment and action seeds for each episode. The upstream wrapper repeats each command for two native suite steps. The score and strict hit flag use wrapped qpos at the Gym action boundary. The original task's visual target is hidden by the simulator configuration. Episode-level RNG and reset choices are reproduction conventions.

Offline training uses 543,000 overlapping windows, a 90/10 window split, batch 128, AdamW, a one-epoch warm-up, cosine decay, the LeWM latent prediction loss, SIGReg and 20 epochs. The reward head uses a separate episode-disjoint 90/10 split on original-dynamics score labels. `online_adapter.py` supplies the fixed pretrained model, the frozen reward head and the native environment to the shared planning and adaptation runner. `checks/` holds directly measured validation products.
