# Reacher action replays

The [arm replay archive](https://github.com/ZhengQingJing/jepa-ttt-reproduction/releases/download/v1.1.0/arm-replays.zip) contains frozen and adapted policies filmed side by side from the same held-out reset. Extract it at the repository root to use the commands below. Each clip directory includes `paired.mp4`, `final.png`, `replay.json` and `actions.npz`.

| Disturbance | Terminal checkpoint | Representative clip | Success-subgroup clip |
| --- | --- | --- | --- |
| Joint phase | Deployment 2, episode 500 | `reacher/joint-phase/e500-median` | `reacher/joint-phase/e500-success` |
| Harmonic | Deployment 1, episode 500 | `reacher/harmonic/e500-median` | `reacher/harmonic/e500-success` |

Paths in the table are relative to the extracted `arm-replays/` directory. Each disturbance also has a representative episode-50 clip. The representative deployment has a final mean score closest to the three-deployment average. Within each deployment the representative start has a paired score gain closest to the median across 100 held-out starts. The success subgroup applies the same selection rule to starts where only the adapted controller reached the target.

Both video checkpoints at episode 500 match the original predictor hashes. Scores, step counts and hit flags for the recorded starts match the corresponding formal evaluations. Each JSON also records the verification performed at capture time against raw joint positions and native camera frames.

## Replay recorded actions

Use the [Reacher environment setup](../README.md). From the repository root:

```bash
export MUJOCO_GL=egl
python experiments/reacher/replay/replay_actions.py arm-replays/reacher/joint-phase/e500-median
python experiments/reacher/replay/replay_actions.py arm-replays/reacher/harmonic/e500-success --frames runs/reacher-frames
```

The script executes the saved commands in the pinned simulator and checks joint positions at every step plus terminal scores and goal hits. With Pillow installed, `--frames` writes camera-0 RGB images into `frozen/` and `adapted/` subdirectories. The goal reference in the distributed video shows joint angles (0, 0).
