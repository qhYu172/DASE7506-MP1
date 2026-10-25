# Public-method study and independent implementation (25 October 2026)

The official submission website currently seals artifact URLs for the instructor. Public GitHub repository search found the repositories below independently; their relationship to a particular sealed submission has not been established. Reported scores in their READMEs are author claims, not locally reproduced scores.

## Sources consulted

- [CaspianW/dase7506-mp2](https://github.com/CaspianW/dase7506-mp2), tree `a1973666c71c4ba72a1183b337043f8fb050c06e`: README, answer-group representation code and question-number features. Ideas: pool candidates sharing a numeric answer; compare groups; weight questions equally; training-only nonempty subsets within an answer group. The subset-augmentation concept is reimplemented independently in `train_augmented_ranker.py`.
- [vectorBH6/DASE7506-MP2](https://github.com/vectorBH6/DASE7506-MP2), tree `00338e6d9f65747ade4998a268930633e1d934d0`: group-ranker training utilities. Ideas: listwise answer-group supervision, question-level folds and training-only normalization.
- [Lyan9145/DASE7506-MP2](https://github.com/Lyan9145/DASE7506-MP2), tree `11f6f0437dd9a3284e39ba54bc239fa850e9557e`: README and NN model. Ideas considered: DeepSets group context and listwise supervision. This neural architecture is not incorporated.
- [CARDINAL-02/DASE7506-MP2-Submission](https://github.com/CARDINAL-02/DASE7506-MP2-Submission), tree `7018785f947ca081cc46fceddf93aad2fa926ac4`: arithmetic/coverage features and portable tree inference. Ideas considered; this code is not incorporated.

The inspected trees did not expose a repository-wide code license. Accordingly, no source file or trained checkpoint was copied into this project. The new files `code/answer_ranker.py`, `code/train_answer_ranker.py` and `code/train_augmented_ranker.py` independently implement the cited general ideas, with explicit attribution. Existing self-written surface features are pooled by mean and range. The initial group softmax was compared with a pairwise logistic objective using training-only within-answer subsets. Additional numeric plausibility features include integer status, magnitude relative to question numbers, denominator size and nearest-number distance. Hyperparameters are chosen by question-level train-only cross-validation, then validation is measured. No test predictions or reference answers from other projects are used.

## Measured adaptation results

All CV figures use the same three folds of 2,114 conflicting-answer train questions (seed 7506), refitting normalization within each fold.

| Independent variant | Best train CV | Validation hard |
| --- | ---: | ---: |
| Group softmax, pooled features | 80.75% | 298/355 = 83.94% |
| Pairwise subsets, pooled features | 81.74% | 297/355 = 83.66% |
| Pairwise subsets, pooled + numeric features | **82.31%** | **303/355 = 85.35%** |

Equal probability fusion of the last model with our v2 gave 299/355, below the standalone new model; it was rejected on validation. The selected independent model was frozen before final test scoring and achieved **390/467 = 83.5117773019272%**, versus v2's 372/467. The inference checkpoint occupies 28,205 bytes including configuration. This is an evidence-backed adaptation, not a claim that the conceptual ingredients were invented here or that the public source implementations were reproduced.

Reproduction: `code/train_answer_ranker.py`, `code/train_augmented_ranker.py`, `code/select_fusion.py`; detailed machine-readable records under `code/runs/answer_ranker/`, `augmented_ranker/`, and `fusion_v3/`. The final inference bundle is `code/runs/model_bundle_v3/`.

This is substantive AI-assisted implementation and research; retain this disclosure with the submission.
