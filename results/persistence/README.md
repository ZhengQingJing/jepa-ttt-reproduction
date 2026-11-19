# Matched persistence control

This experiment reproduces the Two-Room persistence ablation in Appendix A6.1, Table A5 of [JEPA-TTT](https://arxiv.org/pdf/2610.00722#page=29). Both arms start from the same pretrained model and frozen reward head. Each runs on its own CEM trajectories with the same deployment and evaluation seed sequences.

| Dynamics shift | Source | Persistent best | Reset best | Persistent AUC | Reset AUC |
|---|---|---:|---:|---:|---:|
| Grid rotation | Paper | 0.7892 ± 0.0157 | 0.2178 ± 0.0061 | 0.7189 ± 0.0193 | 0.2127 ± 0.0043 |
| Grid rotation | [This reproduction](grid/comparison.json) | 0.7308 ± 0.0120 | 0.1761 ± 0.0041 | 0.6513 ± 0.0323 | 0.1710 ± 0.0002 |
| Spatial wave | Paper | 0.7971 ± 0.0016 | 0.5972 ± 0.0135 | 0.7692 ± 0.0041 | 0.5614 ± 0.0089 |
| Spatial wave | [This reproduction](wave/comparison.json) | 0.7301 ± 0.0195 | 0.4641 ± 0.0259 | 0.6574 ± 0.0413 | 0.4503 ± 0.0036 |

Retaining state increased normalized AUC by 0.4803 on grid rotation and 0.2071 on spatial wave. The two arms used the same update budget in all three paired runs per shift.

The **persistent** arm carries the dynamics predictor and AdamW state into the next deployment episode. The **reset** arm reloads the initial dynamics predictor and starts a fresh AdamW at the beginning of every deployment episode. All episode-local windows are cleared at the boundary in both arms.

Our implementation interprets *dense windows from the current episode* as a chronological stream. A 50-step deployment episode produces 32 valid windows. We process a full batch of eight new windows at a time, with four optimizer updates per episode. All 500 deployment episodes run to the fixed horizon. At episodes 0, 50, …, 500, the current weights are evaluated without updates on the same 100 held-out initial conditions. For reset, evaluation uses the just-adapted end-of-episode weights before the next episode resets them. The paper does not specify the exact minibatch sampling order or the evaluation boundary order for this ablation.

Run once per choice of shift, arm, and seed, with the scratch world model and reward head from the common offline training pipeline:

```bash
python src/persistence.py --shift grid --arm persistent --run 0
python src/persistence.py --shift grid --arm reset --run 0
```

The other deployment seeds are 1 and 2. Use `--shift wave` for the spatial-wave shift. After all three paired streams finish for a shift, aggregate its evaluation curves:

```bash
python src/summarize_persistence.py runs/persistence --shift grid
```

The reported best score is selected from the curve averaged over deployment seeds at episodes 50–500. Its standard deviation is calculated across the three scores at that selected evaluation point. Normalized AUC uses the trapezoid rule separately on each run over 0–500 episodes. All uncertainty values are sample standard deviations over deployment seeds, conditional on a single pretraining checkpoint.
