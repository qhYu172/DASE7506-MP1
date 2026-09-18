# DASE7506 Mini Project 1 — Final Language Model Submission

**Frozen full-test result:** **1.6091548 bits per byte (BPB)**, with **19.471 seconds** of FP32 CPU scoring time. The matching [checkpoint](code/runs/iter_13/checkpoint.pt) is included in this repository, so evaluation does not require retraining. The three-page [academic report](REPORT.pdf) is also available as [Markdown](REPORT.md) and [LaTeX source](REPORT.tex); [EXPERIMENT.md](EXPERIMENT.md) records all 13 iterations.

| Resource or result | Final submission | Limit / reference |
|---|---:|---:|
| Full-test BPB | **1.6091548** | Baseline: 2.1012604 |
| CPU scoring time on recorded test run | **19.471 s** | 5× the same-host baseline test time of 14.644 s = 73.218 s |
| Peak evaluation RAM | **<4 GiB reported**; separate sampled validation peak: 1.514 GiB | 4 GiB |
| Checkpoint size | **42,412,469 bytes (40.448 MiB)** | 64 MiB uncompressed assets |
| Validation BPB | **1.5943168** | Used for selection, not the reported test score |

Scoring time depends on CPU and workload. The final predictor requests up to 32 PyTorch CPU threads on the recorded 32-logical-CPU host. The test JSON does not itself include a peak-RAM field; the validation RAM figure is a sampled process-tree working-set estimate.

## Method in brief

1. **Scale within the asset budget.** The decoder has 10,596,160 parameters: eight attention blocks, width 320, ten heads, a 1,280-unit MLP, RMSNorm, and tied token/output embeddings. It was trained from scratch for 2,400 steps with seed 17.
2. **Discrete exact-match causal cache.** At each position, the predictor searches preceding positions for the longest available matching trigram, bigram, or unigram, weights their already-observed successors by `exp(-(t-j)/512)`, and mixes this distribution with the decoder. All matches obey `j < t`, and cache state is local to each independent call. This adds no learned parameters or neural matrix multiplications, although matching and accumulation do require CPU work.
3. **Post-training temperature calibration.** Validation-only selection fixed the decoder softmax temperature at **1.10** before the final test evaluation. The cache rules and trained weights did not change.

An **architectural diet** replaced the earlier SwiGLU MLP with a two-linear-layer GELU MLP and RoPE with learned absolute positions. The earlier candidate needed 39.165 s for a four-thread validation pass and failed its CPU gate. Under a matched 32-thread validation setting, the earlier model took 25.182 s and the final predictor took 16.948 s. The final *test* run took 19.471 s; those validation and test times are different workloads and should not be treated as a controlled before/after speed measurement. The architecture, model size, and thread setting also changed, so the experiment cannot isolate the effect of either architectural substitution.

The final-checkpoint inference-only ablation uses the **same trained weights** for all variants:

| Cache | Temperature | Full-validation BPB |
|:---:|---:|---:|
| Off | 1.00 | 1.643535445 |
| Off | 1.10 | 1.635770970 |
| On | 1.00 | 1.601842055 |
| On | 1.10 | **1.594316786** |

Run `python ablate_final.py` from `code/` after installing the dataset to reproduce the [ablation record](code/runs/iter_13/ablation_validation_cpu_fp32.json). This uses the official validation scoring function and does not retrain or modify the frozen predictor. CPU timings can vary with host load; the later ablation session was slower than the frozen scoring audit.

## Reproduce the frozen result

Use Python 3.12 and the original, unmodified course dataset and tokenizer. The dataset is intentionally **not redistributed**. Obtain the course package and place its `data/` directory at `code/data/`, including its `manifest.json`, `tokenizer.json`, and the three WikiText split files. `common.py` checks the supplied data hashes. The complete test file is used only for final scoring; model and temperature selection used validation.

From the repository root in Bash:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
cd code
python -m unittest discover -s tests -v
python evaluate.py --checkpoint runs/iter_13/checkpoint.pt --device cpu --precision fp32 --split validation --output runs/iter_13/reproduced_validation.json
python evaluate.py --checkpoint runs/iter_13/checkpoint.pt --device cpu --precision fp32 --split test --output runs/iter_13/reproduced_test.json
```

The reported test output is [test_cpu_fp32.json](code/runs/iter_13/test_cpu_fp32.json). To retrain instead of using the supplied checkpoint (about 44 minutes of training excluding intermediate validation on the recorded host), run from `code/`:

```bash
python train.py --implementation student --device cpu --seed 17 --eval-every 200 --threads 32 --run-dir runs/reproduction
```

The trainer defaults to 2,400 updates, batch size 32, a 100-step warmup, peak learning rate `0.001`, minimum `0.0001`, and a 3,600-step cosine horizon. It processes 19,660,800 training targets. Training from scratch may differ across hardware or software versions; the supplied checkpoint is the artifact for reproducing the official result.

Verify the frozen artifacts from the repository root:

```bash
sha256sum code/runs/iter_13/checkpoint.pt code/student.py
```

Expected SHA-256 values: checkpoint `aa653489ff62d0101589027a915ed6a815e68d1d3c3b0d0a4bf2bda4e94db9ab`; predictor source `ac10afb46a56df4ba787f5fc05379dd5b9effd4efce904e5781cc62176a67c12`.

To rebuild the report PDF, run `pdflatex -interaction=nonstopmode REPORT.tex` where a LaTeX engine is available. As a fallback, install `reportlab` and run `python build_report_pdf.py` from the repository root; the committed PDF was produced through this fallback and visually checked at three A4 pages.

## Reproducibility and attribution

The scorer uses protocol `7506-mp1-wt2-v2`, independent 256-token windows, the course BPE-2048 tokenizer, and full-split UTF-8 byte counts. The final test run covers 428,405 scored targets and 1,292,013 bytes. Source, training recipe, baseline implementation, tests, final configuration, checkpoint, and final-run JSON are committed here. Earlier checkpoints and raw data are excluded to keep the public bundle focused and to respect the dataset redistribution constraint.

WikiText-2 was introduced by [Merity et al.](https://arxiv.org/abs/1609.07843); the underlying text is from Wikipedia contributors. Substantive AI assistance was used to implement experiments, run audits, analyze results, and draft this documentation. The author is responsible for checking the submitted code, figures, and claims.
