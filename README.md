# DASE7506 Mini-project 2: Selecting Reliable LLM Reasoning

**Yu Qinghao | Student ID 3036843599**

Final method **v11**: answer-group ranking with arithmetic checking and propagation of explicit arithmetic slips. Official complete-test accuracy: **85.01070663811564% (397/467)**. Validation: **751/798**.

- [Final report (5 pages)](output/pdf/MP2_report.pdf)
- [Final checkpoint](code/runs/experiment_v11/model.npz)
- [Official test score](code/runs/test-v11-score.json)
- [Frozen source and asset hashes](code/runs/experiment_v11/freeze.json)
- [Final results and evidence index](RESULTS_V11.md)
- [Reuse disclosure](code/runs/v10-sources.json) and [development notes](code/runs/experiment_v11/NOTES.json)

## Evaluate without retraining

Use Python 3.12. From the repository root:

```powershell
cd code
py -3.12 -m pip install -r requirements.txt
py -3.12 fetch_data.py
py -3.12 check_data.py
py -3.12 evaluate_v11.py
py -3.12 verify_v11.py
```

On Linux/macOS replace py -3.12 with your Python 3.12 executable. The fetcher downloads official data at pinned commit bb801f4f2e7629d270df765ab12c333b2dc616bb and validates hashes. Existing data is reused. After setup, evaluation works offline.

The wrapper verifies frozen files and invokes the unchanged official predictor and scorer for validation and test in fresh serial processes with four numerical threads. Expected counts: **751/798** and **397/467**. Prediction entry: code/repair_v11.py. Checkpoint directory: code/runs/experiment_v11/. Only model.npz is loaded for inference (10,166 bytes). Recorded local complete test evaluation: 1.504 seconds on Intel Core i9-14900HX; reference Xeon runtime unmeasured.

## Reproduce training

Run in a separate checkout to preserve distributed frozen files:

```powershell
cd code
py -3.12 train_augmented_ranker.py
py -3.12 verify_anchor_rebuild.py
py -3.12 train_v5.py
py -3.12 train_v11.py
```

The first two commands rebuild the v3 anchor and verify its numerical equivalence to the shipped prerequisite. train_v5.py regenerates the train-only subset-pair cache (plus historical PCA comparisons). train_v11.py uses question-fold CV with refitted anchors and corrections, selects the correction, applies the validation retention check, and writes runs/experiment_v11. These training scripts never open test references. Only the official scorer reads test references for evaluation.

Earlier scripts and compact experiment records are retained for transparency. RESULTS.md describes historical v3; RESULTS_V11.md describes the current final method. The retained v3 checkpoint is a retraining prerequisite, not the current inference checkpoint. Together v3 and v11 assets total 38,371 bytes.

## Attribution and disclosure

Conceptual reuse is recorded in PUBLIC_METHODS.md and code/runs/v10-sources.json. No third-party source file, checkpoint or prediction file was copied. OpenAI Codex substantially assisted research, implementation, experiments, verification, report drafting and submission preparation. Previous aggregate test scores and public reported scores were known; the report discloses repeated development and does not claim an untouched blind test. No external dataset, additional pretrained weight, LLM/API inference or answer lookup is used.
