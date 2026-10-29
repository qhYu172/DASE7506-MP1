# Final v11 evidence record

Yu Qinghao | 3036843599 | 29 October 2026

Official test: **397/467 = 85.01070663811564%**. Validation: **751/798**. Prediction entry: `code/repair_v11.py`; checkpoint: `code/runs/experiment_v11/model.npz`.

## Evidence index

- Final 28-setting, three-fold CV: `code/runs/experiment_v11/development.json`.
- Pre-test freeze: `code/runs/experiment_v11/freeze.json`.
- Official scores: `code/runs/test-v11-score.json`, `validation-v11-score.json`.
- Resource records: `code/runs/test-v11-resources.json`, `validation-v11-resources.json`.
- Synthetic and validation-input invariance checks: `code/runs/experiment_v11/verification.json`.
- Sources: `PUBLIC_METHODS.md`, `code/runs/v10-sources.json`.
- Chronology: `code/runs/experiment_v11/NOTES.json`, `code/runs/development-2026-10-29.json`.

## Controlled final comparison

| Family | Train-question CV correct / 2114 |
|---|---:|
| v3 anchor | 1740 |
| Local arithmetic | 1759 |
| Propagation | 1750 |
| Combined | **1764** |

Each fold refits the anchor and all normalization/coefficients using training questions only. Seed: 7506. Combined correction selected alpha 0.1 and strength 1. Fold counts improve from [588,584,568] to [591,594,579]. This is selection CV, not nested CV. The selected method passes the declared validation retention threshold of 750/798.

## Reproduction and resources

See README.md for exact evaluation and training commands. The pipeline recreates the anchor with train_augmented_ranker.py, verifies equivalence with verify_anchor_rebuild.py, recreates train pairs with train_v5.py, and refits v11 using train_v11.py. Run training in a separate checkout because it overwrites experimental outputs.

Recorded initial v11 test runtime is 1.5039 seconds for fresh prediction plus fresh independent scoring, using four numerical threads, on Intel Core i9-14900HX. Reference Xeon unmeasured. Final uncompressed model.npz is 10,166 bytes (1,112 anchor plus 94 correction float64 coefficients). The retained v3 training prerequisite adds 28,205 bytes.

V11 was frozen and evaluated once during development; later identical-checkpoint reruns are reproduction only. Earlier aggregate scores were already known: v3 390/467, v9 384/467, v10 394/467. No per-question test errors were inspected. This history and substantive Codex assistance are disclosed in the five-page report.
