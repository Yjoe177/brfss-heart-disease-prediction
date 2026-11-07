# L2 regularization study

Training-side stratified 80/20 split of all 328,135 labeled samples (split seed 42).


This is the same seed-42 80/20 partition used during earlier development. F1 values use thresholds optimized on this same calibration portion; they describe model selection and regularization behavior, not independent generalization.

| lambda | F1 | Precision | Recall | AUROC | PR-AUC | Train loss | Calibration loss | |w| |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4368 | 0.3687 | 0.5358 | 0.8622 | 0.4231 | 0.21818 | 0.22037 | 10.537 |
| 1e-05 | 0.4374 | 0.3700 | 0.5348 | 0.8625 | 0.4232 | 0.21840 | 0.22021 | 7.663 |
| 0.0001 | 0.4360 | 0.3742 | 0.5222 | 0.8623 | 0.4157 | 0.21960 | 0.22069 | 4.180 |
| 0.001 | 0.4284 | 0.3407 | 0.5770 | 0.8581 | 0.3897 | 0.22332 | 0.22394 | 2.060 |

Best calibration F1: lambda = 1e-05.
