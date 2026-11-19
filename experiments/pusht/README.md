# PushT

This task implements the two PushT shifts and the original coverage objective from [JEPA-TTT, Appendix A2.1](https://arxiv.org/pdf/2610.00722#page=20). Each command is an absolute target in a 512-pixel arena, represented by two values scaled to [−1,1]. The circular agent pushes a T-shaped body toward the visible green target at (256,256), angle π/4. The reported episode score is the peak fraction of target area covered by the movable block. The strict reach criterion used here is 95% overlap, the coverage threshold declared by the upstream environment.

- **Action rotation** rotates the physical target 90° about the current pusher position before native PD control, following equation (6).
- **Contact rotation** applies native agent control. In each physics substep, it rotates the block's translational displacement by 90° when the pusher and block are in contact, including the immediately following release substep for impulse integration. Angular motion and uncontacted displacement use the upstream solver. This is our implementation of the paper's substep convention.
- **Collection** draws a target with equal probability from uniform arena coordinates or a Gaussian around the block (60 px standard deviation). A sampled command is held for four simulation steps. The source dataset uses 3,000 episodes in unshifted dynamics, terminating at strict success or 300 steps. One native RGB observation and coverage value accompany each state, including the final state. Planning receives RGB observations and issued actions.

From this folder after the repository's standard environment setup:

```bash
python -m pip install -r requirements.txt
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
python collect.py --output data/offline --workers 6
python verify.py --data data/offline --output runs/data-verification.json
python train_offline.py pack
python train_offline.py train
python train_offline.py head
```

The packed native RGB array is memory-mapped from disk. A full 900,000-action dataset takes about 136 GB of uncompressed storage in addition to the compressed HDF5 shards. Pretraining uses 20 epochs, batch 128, three context frames, frame skip five, latent prediction and SIGReg, mixed bfloat16 precision, and epoch-boundary optimizer checkpoints. The model's offline reward head learns coverage from source-only labels with an episode-disjoint 90/10 split.

```bash
for shift in action contact; do
  python ../../src/benchmark.py frozen --task pusht --shift "$shift" --adapter adapter.py --output runs
  for seed in 0 1 2; do
    python ../../src/benchmark.py ttt --task pusht --shift "$shift" --adapter adapter.py --output runs --seed "$seed"
  done
done
```

Each deployment uses 500 full 300-step episodes, peak coverage evaluation at 100 fixed held-out starts every 50 episodes, and a separate fixed 100-trajectory behavior-policy latent prediction test. The 301 observations from a deployment episode support 282 dense windows; the update count after 500 episodes is 2,203. The planner uses the paper's PushT reward scale of 10. Multiplying rewards by a positive constant preserves CEM candidate ordering in exact arithmetic.

## Source dataset measurements

The [collection manifest](manifest.json) records 900,000 commanded actions and 903,000 RGB frames. Three complete episodes replayed with bit-identical RGB, matched simulator states, and coverage errors below 0.000001. The reward-head validation split illustrates the low frequency of high-coverage states.

| Held-out source frames | Paper | Reproduction |
|---|---:|---:|
| Total | 90,300 | 90,300 |
| Coverage below 0.1 | 86,466 | 86,084 |
| Coverage at least 0.5 | 114 | 112 |

The score-distribution audit uses offline frame labels. [Native geometry and data replay](data-verification.json) · [Paper Appendix A6.3](https://arxiv.org/pdf/2610.00722#page=30)

An independent [GPU interface check](interface-smoke.json) used miniature weights trained on three source episodes for one epoch. For both action and contact rotation, the common frozen/TTT driver processed a fixed 300-step diagnostic, then performed four optimizer updates on 282 dense windows in one online episode. The predictor fingerprint changed and the frozen representation matched its initial fingerprint.
