# OGBench-Cube action replays

The [arm replay archive](https://github.com/ZhengQingJing/jepa-ttt-reproduction/releases/download/v1.1.0/arm-replays.zip) contains matched native front-camera recordings for the UR5e hover task. Extract it at the repository root. Under `arm-replays/cube/{quadratic,delayed}/` the `e50-median`, `e150-median` and `e500-median` directories contain `paired.mp4`, `final.png`, replay metadata and separate frozen and adapted command traces.

The filming checkpoints come from a repeat of deployment seed 0 under each disturbance. Both episode-500 hashes differ from the corresponding original deployment hashes. Each final checkpoint was therefore evaluated again on all 100 held-out starts and the frozen baseline was rechecked. For each disturbance the final video uses the start closest to the median paired score gain in this new evaluation. The raw repeat evaluations are in `independent-eval-100.json`. The original three-deployment results are in [the formal result directories](../../../results/additional/cube).

| Disturbance | 100-start repeat score, frozen → adapted | 100-start repeat success, frozen → adapted | Filmed start | Adapted first hit |
| --- | ---: | ---: | ---: | ---: |
| Quadratic crosstalk | 0.2063 → 0.6379 | 26% → 100% | 1000041 | Step 20 |
| Delayed cyclic actuation | 0.0571 → 0.6427 | 0% → 100% | 1000049 | Step 18 |

The episode-150 quadratic clip has an adapted score of 0.2265, compared with 0.2689 at the same archived start. Each clip's `replay.json` identifies the selection reference, observed score and original archived score. The video marks the projected target and end effector for the viewer. The hit criterion uses their real three-dimensional distance.

## Replay recorded actions

Use the [Cube environment setup](../README.md). From the repository root:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python experiments/cube/replay/replay_actions.py arm-replays/cube/quadratic/e500-median
python experiments/cube/replay/replay_actions.py arm-replays/cube/delayed/e500-median --check-pixels --frames runs/cube-frames
```

The script replays both traces and verifies the transformed commands, joint state, 3-D end-effector position, score and hit flag. `--check-pixels` also compares native RGB bytes with the recorded per-frame hashes. This strict check depends on renderer and driver reproducibility. With Pillow installed, `--frames` saves unannotated native RGB images for each branch.
