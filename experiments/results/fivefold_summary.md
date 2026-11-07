# Full-data stratified five-fold CV

All 328,135 labeled examples; held-out folds of 65,627 rows (5,795 positives, 59,832 negatives).


Retrospective evaluation of a frozen procedure: all labeled rows informed earlier development and the folds are correlated.

| Model | F1 mean +/- sample SD | AUROC | PR-AUC |
| --- | ---: | ---: | ---: |
| majority | 0.0000 +/- 0.0000 | 0.5000 | 0.0883 |
| ridge | 0.4272 +/- 0.0027 | 0.8583 | 0.4161 |
| logistic_numeric | 0.4022 +/- 0.0036 | 0.8411 | 0.3502 |
| logistic_encoded | 0.4315 +/- 0.0053 | 0.8619 | 0.4188 |
| reg_logistic_encoded | 0.4303 +/- 0.0060 | 0.8622 | 0.4188 |

Paired differences (sample SD over five folds):

- F1(reg_logistic_encoded) - F1(logistic_encoded): -0.0012 +/- 0.0029
- F1(logistic_encoded) - F1(logistic_numeric): +0.0293 +/- 0.0043
- F1(logistic_encoded) - F1(ridge): +0.0043 +/- 0.0051
