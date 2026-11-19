# Additional paper comparisons

Values are recomputed from the three complete runs linked in each row. ± is the sample standard deviation. Episodic AdaJEPA uses the same episode-average estimate for best score and AUC. [Paper tables](https://arxiv.org/abs/2610.00722).

| Task / shift / method | Paper best | Reproduced best | Paper AUC | Reproduced AUC | Paper final MSE | Reproduced final MSE |
|---|---:|---:|---:|---:|---:|---:|
| [cube/quadratic](additional/cube/quadratic) | 0.6280 | 0.6405 ± 0.0116 | 0.5300 | 0.5220 ± 0.0040 | 0.24754 | 0.1501 ± 0.0128 |
| [episodic-baseline/grid/trainable](additional/episodic-baseline/grid/trainable) | 0.2031 | 0.2115 ± 0.0045 | 0.2031 | 0.2115 ± 0.0045 | — | — |
| [episodic-baseline/grid/frozen](additional/episodic-baseline/grid/frozen) | 0.2470 | 0.2015 ± 0.0011 | 0.2470 | 0.2015 ± 0.0011 | — | — |
| [stream-controls/grid/sparse](additional/stream-controls/grid/sparse) | 0.7860 | 0.6707 ± 0.0054 | 0.6620 | 0.5580 ± 0.0057 | — | — |
| [stream-controls/grid/matched-sparse](additional/stream-controls/grid/matched-sparse) | 0.7940 | 0.7375 ± 0.0051 | 0.7170 | 0.6349 ± 0.0370 | — | — |
| [episodic-baseline/wave/trainable](additional/episodic-baseline/wave/trainable) | 0.5791 | 0.4264 ± 0.0032 | 0.5791 | 0.4264 ± 0.0032 | — | — |
| [episodic-baseline/wave/frozen](additional/episodic-baseline/wave/frozen) | 0.5850 | 0.4171 ± 0.0055 | 0.5850 | 0.4171 ± 0.0055 | — | — |
