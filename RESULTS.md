# DASE7506 Mini-project 2: reproduction record

**Student: Yu Qinghao | Student ID: 3036843599**

## Current final version: v3 (25 October 2026)

The final method is now an independently implemented **numeric-answer-group pairwise ranker**, informed by public methods described and credited in [PUBLIC_METHODS.md](PUBLIC_METHODS.md). It pools embeddings and existing surface features by parsed answer, adds numeric plausibility descriptors, and trains on within-answer subsets with equal question weight. Three-fold train-only CV selects the feature set and regularization. The final checkpoint is **`code/runs/model_bundle_v3`**. Historical v1/v2 assets below are not used for v3 inference.

| Measure | v2 | Final v3 |
| --- | ---: | ---: |
| Validation, all 798 | 740 / 798 = 92.73% | 746 / 798 = 93.48% |
| Validation, 355 conflicting-answer groups | 297 / 355 = 83.66% | 303 / 355 = 85.35% |
| Official complete test | 372 / 467 = 79.66% | **390 / 467 = 83.5117773019272%** |

V3 complete local prediction plus independent scoring: **1.265 seconds**, including startup/imports/loading in fresh processes. Inference assets: **28,205 bytes**. SHA-256 of `ranker.npz`: `5c5c0f4a560c1f42acfb27c82abe9dbaacb75090dc8c9c18cf5be9ef33285723`. `code/runs/v3-freeze.json` records the exact matching source and assets before v3 test scoring; `test-v3-score.json` and `test-v3-resources.json` record results.

### Reproduce the final result (PowerShell, from `code/`)

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 check_data.py
py -3.12 evaluate_frozen.py --split validation
py -3.12 evaluate_frozen.py --split test
```

The wrapper checks frozen hashes, sets four numerical threads and runs both official tools. Equivalent direct commands:

```powershell
$env:OMP_NUM_THREADS='4'; $env:OPENBLAS_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
py -3.12 predict.py --method submission.py --checkpoint runs/model_bundle_v3 --split test --output runs/reproduced-v3-predictions.json
py -3.12 score_predictions.py --split test --predictions runs/reproduced-v3-predictions.json --output runs/reproduced-v3-score.json
```

To regenerate v3 training/search records on train/validation: `py -3.12 train_augmented_ranker.py`. It writes `runs/augmented_ranker/ranker.npz`; evaluate that directory with `--method answer_ranker.py`. This does not overwrite the frozen final checkpoint. The complete nine-setting, three-fold search plus full fit took 53.66 seconds locally. The earlier listwise comparison is reproduced by `train_answer_ranker.py`; `select_fusion.py` reproduces the negative fusion comparison. Neither training script reads test inputs or references.

V1 and v2 aggregate test scores were already known before this follow-up, as were public authors' self-reported scores. V3 was selected using train/validation and frozen before its test run; no per-question test errors were inspected or used. The repeated development history means this is not an untouched blind-test study. V3 has not been tuned after its test score.

The final report is `output/pdf/MP2_report.pdf` (generated from `make_report_v3.py`).

---

## Historical v1/v2 record

## Environment and method

- Python 3.12; NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0.
- Training seed: 7506. Training uses only `train` inputs and references. Validation is used to choose hyperparameters. The test reference is used only by the official final scorer after freezing the model.
- Hardware for local timing: Intel Core i9-14900HX, Windows. Four-thread environment variables were set for each library. The official 30-second limit is determined on the reference Xeon, not this laptop.
- Final model (v2): histogram gradient boosting on candidate text, answer agreement and embedding similarity features, fused with a linear classifier and a small 64-unit MLP on the supplied 1024-dimensional embeddings. The validation-selected configuration is stored in `code/runs/model_bundle_v2/training.json`.

## Commands (PowerShell, from `code/`)

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 check_data.py
$env:OMP_NUM_THREADS='4'; $env:OPENBLAS_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
py -3.12 train.py --output runs/model_bundle
py -3.12 train_v2.py
py -3.12 predict.py --method submission.py --checkpoint runs/model_bundle_v2 --split validation --output runs/validation-v2-predictions.json
py -3.12 score_predictions.py --split validation --predictions runs/validation-v2-predictions.json --output runs/validation-v2-score.json
py -3.12 predict.py --method submission.py --checkpoint runs/model_bundle_v2 --split test --output runs/test-v2-predictions.json
py -3.12 score_predictions.py --split test --predictions runs/test-v2-predictions.json --output runs/test-v2-score.json
```

The provided v2 checkpoint allows evaluation without retraining. `diagnostics.py` and `experiment_*.py` are for validation analysis only. The v1 checkpoint and scores are retained as an ablation record.

## Results

| Method | Validation accuracy |
| --- | ---: |
| Most common parsed answer | 86.22% |
| Embedding linear classifier | 89.22% |
| Text/relative/consensus feature boosting, depth 3 | 88.97% |
| Frozen fusion | **91.10%** (727/798) |
| Fusion plus 64-unit MLP (v2) | **92.73%** (740/798) |

On the 355 validation questions with at least two distinct valid parsed answers, v1 reached 80.00% and v2 reached 83.66%. The entire validation set includes 379 questions whose four candidate answers are identical, inflating whole-set accuracy relative to the harder test bank. The v2 method was chosen using validation before its official test run.

The v1 fusion scored **78.37259100642399% (366/467)** on the complete test bank. The frozen v2 method scored **79.65738758029978% (372/467)**, as recorded by the official scorer in `code/runs/test-v2-score.json`. The complete local v2 prediction-plus-scoring time was **2.99 seconds** on a warm local run. Its inference assets total **1,079,586 bytes** (1.030 MiB). The SHA-256 of v2 `model.joblib` is `e7a8a22a1f77d4c6e9a8fbe18024a0f2fc4260a933a2d49c4e0d0e08dffd7aae`.

## Disclosure and limitations

OpenAI Codex provided substantive help with feature design, implementation, experiments, debugging, and report drafting. The assignment's supplied data, parser, runner, and scorer remain unchanged. No external dataset, pretrained weights, or API are used by the method. Surface text features and fixed embeddings do not constitute full arithmetic or semantic verification; the report examines this limitation.
