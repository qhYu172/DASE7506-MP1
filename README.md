# DASE7506 Mini-project 2: Selecting Reliable LLM Reasoning

**Yu Qinghao — Student ID 3036843599**

Final method **v3**: numeric-answer-group pairwise ranking with training-only subset augmentation. Official complete-test accuracy: **83.5117773019272% (390/467)**. The supplied data, numeric parser and scorer are unchanged.

- [Report (3 pages)](output/pdf/MP2_report.pdf)
- [Results, experiment history and exact commands](RESULTS.md)
- [Public-method sources and independent adaptation disclosure](PUBLIC_METHODS.md)
- [Frozen checkpoint](code/runs/model_bundle_v3/)
- [Official test score](code/runs/test-v3-score.json)
- [Source/checkpoint freeze manifest](code/runs/v3-freeze.json)

## Reproduce without retraining

Use Python 3.12. From the repository root:

```powershell
cd code
py -3.12 -m pip install -r requirements.txt
py -3.12 fetch_data.py
py -3.12 check_data.py
py -3.12 evaluate_frozen.py --split validation
py -3.12 evaluate_frozen.py --split test
```

On Linux/macOS, replace `py -3.12` with your Python 3.12 executable. The evaluation wrapper sets four numerical threads, verifies frozen hashes, and launches the official predictor and independent scorer as fresh processes. Expected test result: **390/467**. The checkpoint is included; no model download or retraining is needed. Final inference assets total **28,205 bytes**. Local fresh-process prediction plus scoring took **1.265 seconds** on an Intel Core i9-14900HX; the reference Xeon determines official runtime eligibility.

The GitHub edition omits the large original data arrays. `fetch_data.py` downloads only the official course data at commit `bb801f4f2e7629d270df765ab12c333b2dc616bb` and verifies every file against the supplied manifest. An existing complete local copy is verified and reused. Data license/attribution files are retained under `code/data/`. After data setup, evaluation works offline.

## Reproduce training and comparisons

```powershell
cd code
py -3.12 train_augmented_ranker.py
```

This performs the final three-fold train-question CV search and fits the chosen model using train only. It writes `runs/augmented_ranker/`, leaving the frozen final checkpoint unchanged. Validation is measured for development; test is never opened by this training script. Use `--method answer_ranker.py --checkpoint runs/augmented_ranker` with the official predictor to evaluate the rebuilt checkpoint.

The earlier group-softmax comparison is in `train_answer_ranker.py`. Rebuild the historical v1/v2 models with `train.py` then `train_v2.py`, and run `select_fusion.py` for the reported negative fusion comparison. Historical v1/v2 learned assets are included in the full local project ZIP, but need not be included in the GitHub edition; their training code and score records are retained. Exploratory LightGBM trials use optional `requirements-experiments.txt`.

## Provenance

This project independently implements ideas credited in `PUBLIC_METHODS.md`; other authors' source files and trained checkpoints are not copied. OpenAI Codex substantially assisted with implementation, experiments and report drafting. The report discloses previous aggregate test-score exposure, repeated development and method limitations. No external dataset, additional pretrained weights, LLM/API inference or answer lookup is used.
