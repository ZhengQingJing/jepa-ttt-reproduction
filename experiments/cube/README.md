# OGBench-Cube hover task

This extension implements the reach experiment in Appendix A2.4 of [JEPA-TTT](https://arxiv.org/pdf/2610.00722). It uses the native OGBench single red cube, UR5e and table. The physical cube starts at (0.425, 0, 0.02) metres. The fixed end-effector goal is (0.425, 0, 0.12) metres. The native front camera supplies 224 × 224 RGB images. The task score is exp(−distance / 0.10) and the success threshold is 0.05 metres.

Each agent command contains normalized Cartesian x, y and z displacements. The simulator receives two additional zero coordinates for relative yaw and gripper motion. In the quadratic shift, commands are clipped after adding three times the pairwise products yz, xz and xy to the respective axes. In the delayed shift, the simulator uses the previous command's y, z and x values in that order, starting from zero at reset. Online training uses the agent's original command.

## Data and offline training

Use Linux, a MuJoCo-compatible headless EGL setup and the common JEPA-TTT Python environment. Install the packages in `requirements-cube.txt` and complete `scripts/setup_upstream.py` from the repository root. Run the following from the directory containing these experiment scripts:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python collect.py --workers 4
python train.py pretrain
python train.py reward
bash run_deployment.sh quadratic delayed
```

For a standalone experiment directory, set `JEPA_TTT_ROOT` to the directory containing the pinned reproduction's `src/` and `upstream/`. The collector produces 3,000 episodes of 50 raw simulation steps under the original dynamics. Raw RGB observations, positions and three-dimensional commanded actions are written to memory-mapped arrays in `data/offline`. Allow about 23 GB for this dataset. End-effector position supplies offline labels and evaluation criteria. The model receives RGB and commanded actions during deployment.

## Implementation choices

The paper specifies an equal mixture of guided and Gaussian exploration, including 30% uniform actions in guided episodes. This implementation uses alternating episode types. Gaussian episodes sample each coordinate independently with standard deviation 0.65 before clipping. Guided episodes otherwise use the clipped difference between the fixed target and measured end-effector position divided by 0.05 metres, with independent 0.20 Gaussian command noise. The publication does not specify these noise scales, reset seeds or the gain. We use the native seeded random-arm reset and the full 50-step collection horizon, including interaction after first task success.

The base model is initialized from scratch with a 15-input action encoder for five consecutive XYZ commands. Pretraining uses the common next-latent prediction and SIGReg objectives, 128 windows per batch, a 90/10 shuffled window split, bfloat16 and 20 epochs. Its learning rate peaks at 5 × 10⁻⁵ after a one-epoch warm-up. The frozen visual encoder is used for episode-disjoint reward-head training. The model and reward-head checkpoints are saved under `models/`.

## Checks

Run `python check_task.py` to verify the seeded native renderer, both action transformations and the hover threshold. The [geometric reachability check](verification/geometry.json) records a privileged controller's first target hit at steps 26, 26 and 27 for original dynamics, quadratic crosstalk and delayed actuation. The learned planner never receives that controller or its actions. Run `python verify_collection.py` against a completed collection to check every RGB row, action range and data hash. This reproduction's [collection summary](verification/collection-summary.json) records all 150,000 rows.
