# Episodic AdaJEPA control

This implements the goal-image-free AdaJEPA comparison in Appendix A3 of [JEPA-TTT](https://arxiv.org/abs/2610.00722). It uses the same LeWM initialization, fixed reward head and CEM as the other Two-Room experiments. Each of three separately seeded runs contains 100 episodes. The table reports the run-mean terminal task score and sample standard deviation.

| Shift | Trainable modules | Paper | Reproduction |
|---|---|---:|---:|
| Grid rotation | Last predictor block and encoder projection | 0.2031 ± 0.0101 | 0.2115 ± 0.0045 |

At every episode boundary we restore the initial model parameters. At each completed action block we use the five most recent blocks (or the available prefix), recompute their projected features, and fit next-latent predictions over sliding contexts of at most three keyframes. We initialize Adam for each update and stop gradients through the target. The adapted final transformer block and its final normalization use learning rate 0.0005. The optional visual projection uses learning rate 0.00001. Module mode is evaluation throughout; normalization statistics are fixed. The backbone and reward head stay frozen. These choices specify how the episodic update maps onto the LeWM architecture. [Original AdaJEPA implementation](https://github.com/agentic-learning-ai-lab/adajepa).

Run from the repository root after installing the Two-Room assets:

```bash
for seed in 0 1 2; do
  python src/episodic_baseline.py --shift grid --encoder trainable --seed "$seed"
done
python scripts/check_tworoombaselines.py episodic runs/episodic-baseline/grid/trainable
```

The same entry point accepts `--shift wave` and `--encoder frozen`. To recalculate the published grid row using only the Python standard library, run `python scripts/check_tworoombaselines.py episodic results/episodic/grid/trainable`.
