# Additional paper comparisons

Values are recomputed from the three complete runs linked in each row. ± is the sample standard deviation. Episodic AdaJEPA uses the same episode-average estimate for best score and AUC. [Paper tables](https://arxiv.org/abs/2610.00722).

| Task / shift / method | Paper best | Reproduced best | Paper AUC | Reproduced AUC | Paper final MSE | Reproduced final MSE |
|---|---:|---:|---:|---:|---:|---:|
| [reacher/joint-phase](additional/reacher/joint-phase) | 0.7580 | 0.6351 ± 0.0128 | 0.6190 | 0.3787 ± 0.0202 | 0.48927 | 0.2514 ± 0.0322 |
| [reacher/harmonic](additional/reacher/harmonic) | 0.6880 | 0.7406 ± 0.0876 | 0.5180 | 0.4314 ± 0.0457 | 0.35488 | 0.2782 ± 0.0092 |
| [cube/quadratic](additional/cube/quadratic) | 0.6280 | 0.6405 ± 0.0116 | 0.5300 | 0.5220 ± 0.0040 | 0.24754 | 0.1501 ± 0.0128 |
| [cube/delayed](additional/cube/delayed) | 0.6430 | 0.6452 ± 0.0033 | 0.5130 | 0.4678 ± 0.0052 | 0.10688 | 0.1032 ± 0.0035 |
| [episodic-baseline/grid/trainable](additional/episodic-baseline/grid/trainable) | 0.2031 | 0.2115 ± 0.0045 | 0.2031 | 0.2115 ± 0.0045 | — | — |
| [episodic-baseline/grid/frozen](additional/episodic-baseline/grid/frozen) | 0.2470 | 0.2015 ± 0.0011 | 0.2470 | 0.2015 ± 0.0011 | — | — |
| [stream-controls/grid/sparse](additional/stream-controls/grid/sparse) | 0.7860 | 0.6707 ± 0.0054 | 0.6620 | 0.5580 ± 0.0057 | — | — |
| [stream-controls/grid/matched-sparse](additional/stream-controls/grid/matched-sparse) | 0.7940 | 0.7375 ± 0.0051 | 0.7170 | 0.6349 ± 0.0370 | — | — |
| [stream-controls/grid/dense](additional/stream-controls/grid/dense) | 0.7820 | 0.7386 ± 0.0102 | 0.6860 | 0.5987 ± 0.0208 | — | — |
| [episodic-baseline/wave/trainable](additional/episodic-baseline/wave/trainable) | 0.5791 | 0.4264 ± 0.0032 | 0.5791 | 0.4264 ± 0.0032 | — | — |
| [episodic-baseline/wave/frozen](additional/episodic-baseline/wave/frozen) | 0.5850 | 0.4171 ± 0.0055 | 0.5850 | 0.4171 ± 0.0055 | — | — |
| [stream-controls/wave/sparse](additional/stream-controls/wave/sparse) | 0.5050 | 0.6559 ± 0.0213 | 0.4270 | 0.4464 ± 0.0073 | — | — |
| [stream-controls/wave/matched-sparse](additional/stream-controls/wave/matched-sparse) | 0.7900 | 0.7320 ± 0.0037 | 0.6720 | 0.5994 ± 0.0102 | — | — |
