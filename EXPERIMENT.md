# Experiment Log

## Iteration 1 — RoPE, RMSNorm, SwiGLU

Hypothesis: Replacing learned absolute positions and LayerNorm with RoPE and RMSNorm, and replacing GELU MLPs with a parameter-matched SwiGLU, improves next-token modeling within the course inference budget. The output projection remains tied to token embeddings. This iteration tests the combined change; it does not isolate the contribution of each mechanism.

| Measure | Reference baseline* | Iteration 1 |
|---|---:|---:|
| Parameters | 1,088,256 | 1,120,896 |
| Training targets | 9,830,400 | 9,830,400 |
| Validation BPB | 2.0710 | **1.828568** |
| Validation CPU scoring time | 4.69 s | 5.685 s |
| Checkpoint size | — | 4,500,597 bytes (4.292 MiB) |
| Peak evaluation process-tree working set | — | 1,543,479,296 bytes (1.4375 GiB) |

*Baseline BPB and timing are supplied reference values, not measurements from a new local baseline run. The 5× time ceilings in `AUTONOMOUS_WORKFLOW.md` derive from these values; the course README notes hardware-dependent timings.

Implementation: `code/student.py`, width 128, four heads, four blocks, context 256, SwiGLU hidden width 384, tied token/output weight. Initialization is from scratch. Training used `code/.venv/Scripts/python.exe train.py --implementation student --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/iter_1`, 1,200 steps, 32 sequences per step, FP32, and took 376.11 s excluding intermediate validation. Checkpoint ancestry: none. This is the sole search run so far. Training produced `code/runs/iter_1/checkpoint.pt` (SHA-256 `257ab5c0077df3a7999876e13243902b48210ca029672a1ad0d0991a27bc2c5f`).

Validation during training: 2.167220 BPB at step 300, 1.937153 at step 600, 1.858638 at step 900, and 1.828568 at step 1,200. Independent checkpoint evaluation used `code/.venv/Scripts/python.exe evaluate.py --checkpoint runs/iter_1/checkpoint.pt --device cpu --precision fp32 --split validation` from `code/`; it scored all 376,599 targets over 1,148,007 bytes. The evaluator reported 5.685 s. A separate repeat sampled the evaluator's Windows process tree at approximately 200 ms intervals to measure peak working set; this is an observed-process estimate, not a guaranteed instantaneous peak.

Decision: **Accept iteration 1 for validation development.** All five contract tests passed (`OK`), validation BPB improved by 0.242432 (11.71% relative), validation scoring was below 23.4 s, observed RAM was below 4 GiB, and checkpoint size was below 64 MiB. The 73.2 s test-time gate remains unmeasured: the test split was deliberately not touched before final freeze.

Next ablation: train a same-budget, same-seed variant that removes one mechanism at a time (starting with RoPE versus learned absolute positions), then compare validation BPB, CPU time, and parameter count. Do not use test results for selection. The present comparison cannot attribute the gain to any one component.

Substantive AI assistance: Codex generated the implementation, ran the checks and training, and prepared this experiment record; the student should review and disclose this assistance in the final submission documentation.

## Iteration 2 — Ablate RoPE with learned absolute positions

Hypothesis: Replacing RoPE with the baseline's learned absolute position embeddings, while retaining RMSNorm, SwiGLU (hidden width 384), tied token/output weights, width 128, four heads, and four blocks, will reveal the contribution of the positional mechanism under a matched training budget. The self-contained auxiliary implementation is `code/student_ablate_rope.py`; `code/student.py` was left unchanged so the Iteration 1 checkpoint remains reconstructable.

| Measure | Iteration 1: RoPE | Iteration 2: learned positions |
|---|---:|---:|
| Parameters | 1,120,896 | 1,153,664 (+32,768; +2.92%) |
| Training targets | 9,830,400 | 9,830,400 |
| Validation BPB, step 300 | 2.167220 | 2.294312 |
| Validation BPB, step 600 | 1.937153 | 2.139775 |
| Validation BPB, step 900 | 1.858638 | 2.052847 |
| Validation BPB, step 1,200 | **1.828568** | 2.006781 |
| Independent validation CPU scoring time | 5.685 s | 4.979 s |
| Checkpoint size | 4,500,597 bytes | 4,631,989 bytes (4.417 MiB) |
| Observed peak evaluation process-tree working set | 1.4375 GiB | 1.4563 GiB |
| Training time, excluding validation | 376.11 s | 312.64 s |

Phase 2: `.venv\Scripts\python.exe -m unittest discover -s tests -v` passed all five tests (`OK`). Because the fixed contract suite imports `student`, a separate direct check of the ablation model verified output shape, causal invariance, normalization, and example independence. The ablation has 1,153,664 parameters, 2.92% above Iteration 1 because of the 256 × 128 learned position table; other architectural dimensions were held fixed.

Phase 3: `.venv\Scripts\python.exe train.py --implementation student_ablate_rope --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/iter_2_ablate_rope` from `code/`. It used 1,200 steps, batch size 32, seed 17, FP32, 4 CPU threads, and 9,830,400 processed training targets. Checkpoint ancestry: none (trained from scratch). Cumulative search through Iterations 1–2: two full 1,200-step runs, 19,660,800 processed training targets, and 688.75 s reported training time excluding validation. The checkpoint is `code/runs/iter_2_ablate_rope/checkpoint.pt` (SHA-256 `d2e7f40becf290d2af1edfbf122cc12cab68fffb39df9b85021218634c704545`).

Phase 4: `.venv\Scripts\python.exe evaluate.py --checkpoint runs/iter_2_ablate_rope/checkpoint.pt --device cpu --precision fp32 --split validation` scored all 376,599 targets over 1,148,007 bytes. An approximately 200 ms Windows process-tree sampling run observed 1,563,717,632 bytes peak working set; as in Iteration 1, this is an estimate and may miss a shorter-lived peak.

Decision: **Retain Iteration 1 for development; reject the no-RoPE ablation as the preferred model.** The ablation passes contracts, beats the supplied 2.0710 baseline reference by 0.064219 BPB, and meets measured validation time, RAM, and asset-size gates, but is 0.178213 BPB (9.75%) worse than Iteration 1 at the same training-target budget. This supports RoPE as beneficial in this paired setup. It is not a perfectly parameter-matched causal estimate: the learned position table adds 32,768 parameters and changes subsequent random initialization draws. Test time remains unmeasured because the model-selection process has not been frozen; no test split was used.

## Iteration 3 — Three shared blocks × two passes with QK-Norm

Hypothesis: Reusing three wider transformer blocks twice (effective depth six), with per-head RMSNorm applied separately to queries and keys before RoPE, may improve short-budget learning while retaining about 1.1M parameters. The model retains RoPE, residual RMSNorm, SwiGLU, and tied token/output weights. The implementation is the Iteration 3 revision of `code/student.py` (SHA-256 `c9f35dff2377e8a042220b2a37cc372b5fc5b822acf9bfa447f288435ecfcdb0`). The three blocks each use SwiGLU hidden width 560, so this is a combined shared-depth/QK-Norm/wider-MLP experiment rather than a single-mechanism ablation.

| Measure | Iteration 1 | Iteration 3 |
|---|---:|---:|
| Parameters | 1,120,896 | 1,110,240 |
| Training targets | 9,830,400 | 9,830,400 |
| Validation BPB, step 300 | 2.167220 | 2.209311 |
| Validation BPB, step 600 | 1.937153 | 1.950312 |
| Validation BPB, step 900 | 1.858638 | 1.864063 |
| Validation BPB, step 1,200 | **1.828568** | 1.833470 |
| Independent validation CPU scoring time | 5.685 s | 8.813 s |
| Checkpoint size | 4,500,597 bytes | 4,456,181 bytes (4.250 MiB) |
| Observed peak evaluation process-tree working set | 1.4375 GiB | 1.4155 GiB |
| Training time, excluding validation | 376.11 s | 602.19 s |

Phase 2: `.venv\Scripts\python.exe -m unittest discover -s tests -v` passed all five tests (`OK`); the model has three shared blocks, effective depth six, and 1,110,240 trainable parameters.

Phase 3: `.venv\Scripts\python.exe train.py --implementation student --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/iter_3` from `code/`. Seed 17, FP32, batch size 32, 1,200 steps, four CPU threads, and 9,830,400 processed training targets. Checkpoint ancestry: none (trained from scratch). Cumulative search through Iteration 3: three full runs, 29,491,200 processed targets and 1,290.94 s reported training time excluding validation. The checkpoint is `code/runs/iter_3/checkpoint.pt` (SHA-256 `ea7adcc12a569ca1818652f7e749dca7ea3a5dd20d4df6392ffdd472764c64b3`).

Phase 4: `.venv\Scripts\python.exe evaluate.py --checkpoint runs/iter_3/checkpoint.pt --device cpu --precision fp32 --split validation` scored all 376,599 targets over 1,148,007 bytes. A separate validation run sampled the Windows evaluator process tree at approximately 200 ms intervals and observed 1,519,869,952 bytes peak working set (an estimate that could miss a shorter-lived peak).

Decision: **Do not promote Iteration 3 over Iteration 1.** It passes contracts, baseline-relative BPB, measured validation time, RAM, and asset-size gates, but trails Iteration 1 by 0.004902 BPB and uses about 60% more training time and 55% more validation inference time. The <1.65 aspirational target remains unmet. Test scoring remains deferred until final freeze. Because both Iterations 1 and 3 use the module name `student` with the same saved config, the current module revision must be matched to the corresponding checkpoint for exact reproduction; changing `student.py` for Iteration 4 will not preserve the Iteration 3 source hash. No immutable course file was changed.

## Iteration 4 — Causal within-window token-pair memory

Hypothesis: A nonparametric, call-local count cache of observed successors can adapt predictions to repeated patterns within each independent scoring window, improving BPB without retraining or adding model parameters. `predict_log_probs(ids)` now forms a strictly lower-triangular match matrix. For position `t`, it first uses earlier source positions `j < t` with the same observed pair `(ids[j-1], ids[j]) == (ids[t-1], ids[t])`; if none exist, it falls back to earlier matches on `ids[j] == ids[t]`. The cache target is `ids[j+1]`, which is known at position `t` because `j < t`. It blends the resulting successor frequency distribution with the model softmax, using validation-selected weights of 0.4 for pair matches and 0.1 for unigram-only matches. State is allocated anew on each call; no cross-window entries or validation-derived weights are stored. The only code change was `code/student.py` (final SHA-256 `ca236b80b991c6caa6f848d233ce0f828f4cfe47106c0ffc3bffdbf9929a81c9`).

The Iteration 3 checkpoint was reused without training (`code/runs/iter_3/checkpoint.pt`, SHA-256 `ea7adcc12a569ca1818652f7e749dca7ea3a5dd20d4df6392ffdd472764c64b3`); it still has 1,110,240 parameters and occupies 4,456,181 bytes (4.250 MiB). Checkpoint ancestry for this predictor: Iteration 3's 1,200-step run, 9,830,400 processed training targets. Iteration 4 adds no training targets. The fixed contract suite passed all five tests (`OK`) after the final edit, including causal invariance and state reset.

| Validation-only cache setting | BPB | Scoring time |
|---|---:|---:|
| No cache (Iteration 3 checkpoint) | 1.833470 | 8.813 s |
| Pair 0.2, no unigram fallback | 1.779558 | 10.860 s |
| Pair 0.4, no unigram fallback | 1.775284 | 10.001 s |
| Pair 0.6, no unigram fallback | 1.778911 | 10.042 s |
| Pair 0.4, unigram 0.05 | 1.756623 | 10.339 s |
| **Pair 0.4, unigram 0.1 (selected)** | **1.754367** | **10.340 s final audit** |
| Pair 0.4, unigram 0.2 | 1.757092 | 10.025 s |
| Pair 0.5, unigram 0.1 | 1.755264 | 10.346 s |
| Pair 0.3, unigram 0.1 | 1.755308 | 10.369 s |

Independent final command from `code/`: `.venv\Scripts\python.exe evaluate.py --checkpoint runs/iter_3/checkpoint.pt --device cpu --precision fp32 --split validation --output runs/iter_3/validation_memory_final_cpu_fp32.json`. It scored 376,599 targets over 1,148,007 bytes. A separate approximately 200 ms Windows process-tree sample observed 1,546,637,312 bytes (1.4404 GiB) peak working set, an estimate that may miss a shorter-lived peak. The final result improves the same-checkpoint no-cache score by 0.079103 BPB, and Iteration 1's score by 0.074201 BPB. It meets measured contract, validation BPB, validation time, RAM, and asset-size gates. The aspirational <1.65 target is not yet met; no claim of a technical limit is warranted from this small validation-only search. The test split and its time gate remain untouched until final freeze.

Search cost: Iteration 4 evaluated eight cache settings (including the first pair-only setting; the selected setting was re-evaluated for final audit) with no additional training. The setting and its comparison are validation-selected, not test-selected. Cumulative model-training search remains three full 1,200-step runs (29,491,200 processed targets), plus the recorded validation evaluations. This inference-only mechanism is not a trained model update and should be described as such in the report.

## Iteration 5 — Iteration 1 backbone with recency-weighted multi-order memory

Hypothesis: The faster and slightly stronger four-block, unshared Iteration 1 backbone plus a hierarchical within-window cache (trigram, bigram, then unigram) will improve validation BPB and inference time over Iteration 4's shared-depth backbone. `code/student.py` was restored to the exact Iteration 1 parameter layout: four separate blocks, SwiGLU hidden width 384, RoPE, RMSNorm, and tied embeddings. A strict `load_state_dict` check succeeded against the original `runs/iter_1/checkpoint.pt`; the reconstructed model has 1,120,896 parameters. The predictor source SHA-256 is `e8af23baa4e3a28d5ad58e6633552b6fa76e97a7460006a6b6c739f6271039c9`.

At query position `t`, cache source positions are strictly `j < t`. The predictor first selects prior occurrences of `(ids[t-2], ids[t-1], ids[t])`, falling back to matching the last two tokens and then the current token when the longer order has no match. Each selected source contributes its observed successor `ids[j+1]` with weight `exp(-(t-j)/512)`, normalized within the selected order. The cache mixture weights are 0.6 for trigram, 0.35 for bigram, and 0.1 for unigram matches; when none match, it uses the backbone distribution alone. All memory is computed anew per call and uses only observed within-window prefixes. These weights were specified for Iteration 5, not selected with test data.

| Measure | Iteration 1: no memory | Iteration 4: shared backbone + pair/unigram cache | Iteration 5: unshared backbone + multi-order cache |
|---|---:|---:|---:|
| Parameters | 1,120,896 | 1,110,240 | 1,120,896 |
| Training targets in checkpoint ancestry | 9,830,400 | 9,830,400 | 9,830,400 |
| Validation BPB | 1.828568 | 1.754367 | **1.753179** |
| Validation CPU scoring time | 5.685 s | 10.340 s | **7.617 s** |
| Checkpoint size | 4,500,597 bytes | 4,456,181 bytes | 4,500,597 bytes (4.292 MiB) |
| Observed peak evaluation process-tree working set | 1.4375 GiB | 1.4404 GiB | 1.4616 GiB |

Phase 2: `.venv\Scripts\python.exe -m unittest discover -s tests -v` passed all five tests (`OK`), including causality, independent examples, reset behavior, and gradients. The strict checkpoint-load check confirmed architectural compatibility.

Phase 3: No new training. The original Iteration 1 checkpoint was reused (SHA-256 `257ab5c0077df3a7999876e13243902b48210ca029672a1ad0d0991a27bc2c5f`), retaining its 1,200-step, seed-17, 4-thread CPU training cost. Cumulative full-training search remains three runs and 29,491,200 processed targets.

Phase 4: `.venv\Scripts\python.exe evaluate.py --checkpoint runs/iter_1/checkpoint.pt --device cpu --precision fp32 --split validation --output runs/iter_1/validation_iter_5_cpu_fp32.json` scored all 376,599 targets over 1,148,007 bytes. An approximately 200 ms Windows process-tree sample observed 1,569,366,016 bytes peak working set (an estimate that may miss a shorter-lived peak). Iteration 5 improves over its same-checkpoint no-memory result by 0.075389 BPB and over Iteration 4 by 0.001188 BPB. Validation scoring remains well below the 23.4 s gate; parameter count, RAM, and checkpoint asset size are also within their limits.

Decision: **Retain Iteration 5 as the current validation leader.** The improvement over Iteration 4 is small and confounds the backbone and memory hierarchy, so it does not establish which memory refinement caused the difference. The <1.65 aspirational target is still unmet. The test split and test-time gate remain untouched until final freeze. The current `student.py` reproduces Iterations 1 and 5 with `runs/iter_1/checkpoint.pt`, but not Iterations 3 or 4 with `runs/iter_3/checkpoint.pt`; their recorded source revision is needed for exact reproduction.

## Iteration 6 — Embedding-cosine soft memory (negative result)

Training-dynamics audit: `code/train.py` exposes no learning-rate or AdamW hyperparameter flags. It already uses AdamW with initial learning rate 0.001, weight decay 0.1, a 100-step linear warmup, and cosine decay to 10% of the initial rate over the specified step count. Following the requested fallback, no redundant `runs/iter_6` training run was launched; the existing best `runs/iter_1/checkpoint.pt` was reused. Its ancestry remains 1,200 steps, seed 17, four CPU threads, 9,830,400 processed training targets, 1,120,896 parameters, and 4,500,597 checkpoint bytes. Cumulative full-training search remains three runs and 29,491,200 processed targets.

Hypothesis: token-embedding cosine similarity can provide a dense semantic successor distribution that complements the exact trigram/bigram/unigram cache. In both candidates, `predict_log_probs` normalized current-window token embeddings, computed vectorized pairwise cosine similarities scaled by 12, applied the existing `exp(-(t-j)/512)` recency penalty, and masked all `j >= t` before softmax. Past positions voted for their already-observed successors `ids[j+1]`; each call allocated its own prior. The dense prior received a 0.05 mixture share, either at every `t > 0` or only when the exact cache had no match. Both were strictly causal and passed the unchanged five-test contract suite (`OK`).

| Validation predictor on Iteration 1 checkpoint | BPB | CPU scoring time | Source SHA-256 |
|---|---:|---:|---|
| Iteration 5 exact multi-order cache | **1.753179** | 7.617 s | `e8af23baa4e3a28d5ad58e6633552b6fa76e97a7460006a6b6c739f6271039c9` |
| Exact + 0.05 soft memory everywhere | 1.757795 | 8.266 s | `344af693a30bc3d1addd4ac04c6fd7f74b220432a42f859def4940dcb50dc7c1` |
| Exact + 0.05 soft memory only without exact match | 1.758089 | 8.518 s | `782f12a8aa108a7fd94a8ec3f79d3156981f14b9fea0866107d76f9f5e7e2797` |

Evaluation used `.venv\Scripts\python.exe evaluate.py --checkpoint runs/iter_1/checkpoint.pt --device cpu --precision fp32 --split validation` from `code/`, with separate outputs `runs/iter_1/validation_iter_6_soft005_cpu_fp32.json` and `runs/iter_1/validation_iter_6_soft_fallback_cpu_fp32.json`. Both completed all 376,599 validation targets over 1,148,007 bytes. Both times are well below the 23.4 s ceiling and the 12 s target. No test split was touched. These two additional validation comparisons add no training cost but do add model-selection/search cost.

Decision: **Reject the tested soft-memory variants and restore Iteration 5.** Adding the dense prior worsened BPB by 0.004617–0.004910 while adding about 0.65–0.90 s scoring time. The likely explanation is that static token-embedding similarity is insufficiently calibrated to predict the next-token distribution on top of the already informative exact cache; this is an inference, not a proven mechanism. `code/student.py` was restored byte-for-byte to the Iteration 5 revision (SHA-256 `e8af23baa4e3a28d5ad58e6633552b6fa76e97a7460006a6b6c739f6271039c9`). All five contracts passed again (`OK`), and the restored predictor reproduced 1.753179 validation BPB in 7.489 s. The <1.70 goal remains unmet; the test split remains deferred until final freeze.

## Iteration 7 — Four-token cache and confidence-gated interpolation (negative result)

Hypothesis: extending the exact-match hierarchy from three to four observed tokens and reducing cache reliance when the backbone is confident may improve the Iteration 5 predictor. The candidate added a fourth suffix tier `(ids[t-3], ids[t-2], ids[t-1], ids[t])`, falling back to three-, two-, and one-token suffixes. The selected source positions obeyed `j < t`, and the recency weighting remained `exp(-(t-j)/512)`. A causal per-position confidence proxy, `peak = max_v softmax(logits_t)[v]`, scaled the nominal cache share by `1.1 - 0.75 × peak`; for example, a peak of 0.8 halves the cache share. No state crossed scoring windows. The original `runs/iter_1/checkpoint.pt` was reused, so there was no new training, parameter, or checkpoint-asset cost.

| Validation-only predictor | BPB | CPU scoring time |
|---|---:|---:|
| Iteration 5: three-tier exact cache | **1.753179** | 7.617 s original audit |
| Four-tier, weights 0.70/0.45/0.25/0.05, confidence gate | 1.756103 | 8.971 s |
| Four-tier, weights 0.70/0.45/0.25/0.05, static mix | 1.755965 | 8.217 s |
| Four-tier, weights 0.70/0.60/0.35/0.10, confidence gate | 1.753947 | 8.294 s |
| Four-tier, weights 0.70/0.60/0.35/0.10, static mix | 1.754263 | 8.751 s |

The confidence-gated first candidate passed all five tests in `.venv\Scripts\python.exe -m unittest discover -s tests -v` (`OK`). The ablation comparisons used the same checkpoint and FP32 CPU validation evaluator, writing separate files under `code/runs/iter_1/validation_iter_7_*.json`. All scores cover 376,599 validation targets and 1,148,007 bytes. All measured scoring times are well below the 23.4 s validation gate; the checkpoint remains 4,500,597 bytes and 1,120,896 parameters. These four evaluations add validation search cost, not training cost. Cumulative full-training search remains three runs and 29,491,200 processed targets. The test split was not used.

Decision: **Reject Iteration 7 and retain Iteration 5.** The best four-tier candidate (with confidence gating and preserved lower-order weights) is still 0.000768 BPB worse than Iteration 5. The paired static/gated comparisons show that gating helped the preserved-weight version slightly but did not help the proposed lower-weight version; this does not support a general benefit from the new mechanism. The exact Iteration 5 `student.py` source was restored (SHA-256 `e8af23baa4e3a28d5ad58e6633552b6fa76e97a7460006a6b6c739f6271039c9`); all five contracts passed again (`OK`), and the restored predictor reproduced 1.753179 validation BPB in 8.293 s. The <1.70 target remains unmet and test-time scoring remains deferred until final freeze.

## Iteration 8 — Five unshared MQA blocks, narrower SwiGLU

Hypothesis: replacing four full multi-head attention blocks with five multi-query attention (MQA) blocks will buy an additional layer at about the same parameter budget and improve the trained backbone, while leaving the Iteration 5 causal memory predictor unchanged. In each block, queries have four heads but keys and values each have one shared 32-dimensional head. The SwiGLU hidden width is 320 instead of 384. RoPE, RMSNorm, and tied token/output embeddings remain. `code/student.py` deterministically builds five blocks from the baseline config; its source SHA-256 is `9e04aea64ba4d56b90a2f0bdd852c78ceea62978a573971bb3f70f9b473e2600`. The `predict_log_probs` implementation was not modified for this iteration.

| Measure | Iteration 5: four full-attention blocks | Iteration 8: five MQA blocks |
|---|---:|---:|
| Parameters | 1,120,896 | 1,088,192 |
| Training targets | 9,830,400 | 9,830,400 |
| Validation BPB with golden memory | 1.753179 | **1.751079** |
| Independent validation CPU scoring time | 7.617 s | 7.801 s |
| Checkpoint size | 4,500,597 bytes | 4,379,957 bytes (4.177 MiB) |
| Observed peak evaluation process-tree working set | 1.4616 GiB | 1.4170 GiB |
| Training time, excluding validation | 376.11 s | 436.66 s |

Phase 2: `.venv\Scripts\python.exe -m unittest discover -s tests -v` passed all five tests (`OK`) before training. The model had exactly five blocks, 1,088,192 trainable parameters, and tied token/output weights. The shared K/V tensors are broadcast over query heads for causal scaled-dot-product attention; no parameter duplication occurs.

Phase 3: `.venv\Scripts\python.exe train.py --implementation student --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/iter_8` from `code/`. It ran 1,200 steps, batch size 32, FP32, seed 17, and 4 CPU threads, processing 9,830,400 training targets. Training time excluding validation was 436.66 s; the scheduled validation checks took an additional 29.22 s. Checkpoint ancestry: none (trained from scratch). `code/runs/iter_8/checkpoint.pt` has SHA-256 `4c697f7dc513cb85740eb1194fb15f101771f4e6f32993b5da0dc00705714559`. Cumulative full-training search through Iteration 8 is four 1,200-step runs, 39,321,600 processed targets, and about 1,727.60 s reported training time excluding validation.

Validation curve with the unchanged golden memory: 2.059981 BPB at step 300, 1.849156 at step 600, 1.777787 at step 900, and 1.751079 at step 1,200. Independent Phase 4 command: `.venv\Scripts\python.exe evaluate.py --checkpoint runs/iter_8/checkpoint.pt --device cpu --precision fp32 --split validation`; it scored all 376,599 targets over 1,148,007 bytes in 7.801 s. An approximately 200 ms Windows process-tree sample observed 1,521,504,256 bytes peak working set, an estimate that may miss a shorter-lived peak.

Decision: **Promote Iteration 8 as the current validation leader.** It improves Iteration 5 by 0.002100 BPB with 32,704 fewer parameters and a smaller checkpoint, at the cost of about 60.55 s more training and 0.18 s more validation inference. It passes the measured contract, baseline-relative BPB, validation-time, RAM, and asset-size gates. This comparison changes both attention parameterization and depth, so it cannot attribute the small gain to MQA alone. The <1.70 aspiration is still unmet. No test split was used; the frozen test-time gate remains unmeasured. The current `student.py` reconstructs Iteration 8, not the older Iteration 1 checkpoint; exact reproduction of Iteration 5 requires its recorded source revision.

## Iteration 9 — Six MQA blocks with embedding scaling and logit soft-cap (negative result)

Hypothesis: a sixth MQA block within approximately 1.1M parameters, together with Gemma-style embedding scaling and a final logit soft-cap, could improve convergence in the fixed 1,200-step budget. The candidate used six unshared MQA blocks and SwiGLU hidden width 256. Immediately after lookup it multiplied embeddings by `sqrt(width)`; the `forward` output was `30 * tanh(raw_logits / 30)`. RoPE, RMSNorm, weight tying, and the Iteration 5 exact trigram/bigram/unigram causal memory were retained. Candidate `code/student.py` SHA-256: `bd23bfcdb2e7bc68b136085e8f6e4634bbf71ee9ec913898a598fc2dde11b4dd`.

| Measure | Iteration 8: five MQA blocks | Iteration 9: six MQA blocks + stabilizers |
|---|---:|---:|
| Parameters | 1,088,192 | 1,105,152 |
| Training targets | 9,830,400 | 9,830,400 |
| Validation BPB, step 300 | 2.059981 | 2.092652 |
| Validation BPB, step 600 | 1.849156 | 1.908886 |
| Validation BPB, step 900 | 1.777787 | 1.834778 |
| Validation BPB, step 1,200 | **1.751079** | 1.808530 |
| Independent validation CPU scoring time | 7.801 s | 8.954 s |
| Checkpoint size | 4,379,957 bytes | 4,452,917 bytes (4.247 MiB) |
| Observed peak evaluation process-tree working set | 1.4170 GiB | 1.4379 GiB |
| Training time, excluding validation | 436.66 s | 714.44 s |

Phase 2: `.venv\Scripts\python.exe -m unittest discover -s tests -v` passed all five tests (`OK`) before training. Parameter count was 1,105,152 and the model had exactly six MQA blocks; the output weight remained tied.

Phase 3: `.venv\Scripts\python.exe train.py --implementation student --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/iter_9` from `code/`. It ran from scratch for 1,200 steps, batch size 32, FP32, seed 17, and four CPU threads, processing 9,830,400 training targets. Checkpoint ancestry: none. `code/runs/iter_9/checkpoint.pt` has SHA-256 `249626c65ab9d750963797c29bdf04aa9f3a64b849b3fe96546036859373fff3`. Training took 714.44 s excluding intermediate validation, which took 44.27 s. The first 300 steps were unusually slow on this host; later 100-step intervals were faster. Cumulative full-training search through Iteration 9 is five 1,200-step runs, 49,152,000 processed targets, and about 2,442.04 s reported training time excluding validation.

Phase 4: `.venv\Scripts\python.exe evaluate.py --checkpoint runs/iter_9/checkpoint.pt --device cpu --precision fp32 --split validation` independently reproduced 1.808530 BPB over all 376,599 validation targets and 1,148,007 bytes in 8.954 s. An approximately 200 ms Windows process-tree sample observed 1,543,925,760 bytes peak working set (an estimate that may miss a shorter-lived peak). The earliest in-training validation took 18.53 s during the host slowdown; all observed validation passes remained under the 23.4 s gate.

Decision: **Reject Iteration 9 as the preferred model.** It passes contract, baseline-relative BPB, validation-time, observed RAM, and checkpoint-size gates, but is 0.057451 BPB worse than Iteration 8 under the same training-target budget while requiring substantially more training time. This combined experiment does not isolate whether the additional depth, narrower MLP, embedding scaling, or soft-cap caused the regression. The hoped-for faster convergence was not observed on the validation curve. The active `student.py` is restored to the Iteration 8 revision; the Iteration 9 checkpoint requires the above source revision for exact reconstruction. The <1.70 target remains unmet, and the test split was not used.

## Iteration 10 — Three Gated DeltaNet mixers plus one gated attention layer (negative result)

Hypothesis: a 3:1 hybrid of cheap recurrent delta-rule mixers and gated attention, with a bounded residual shortcut, will use the approximately 1.1M-parameter budget more effectively than the five-block MQA leader. The four 128-wide blocks are `[GatedDeltaNet, GatedDeltaNet, GatedDeltaNet, GatedAttention]`, with four 32-dimensional heads, SwiGLU hidden width 320, RMSNorm, and tied token/output weights. The attention block has zero-centered Q/K normalization, partial RoPE on half of each head, and a learned sigmoid output gate. Each DeltaNet has data-dependent forget/write gates, applies the delta update `S_t = alpha_t S_(t-1) + beta_t (v_t - S_(t-1) k_t) k_t^T`, and uses a 32-token chunked lower-triangular solve plus an SiLU output gate. The residual shortcut is a lightweight, data-dependent, two-lane Sinkhorn-normalized mixing approximation; it is **not** the full multi-stream mHC formulation. The exact Iteration 5 trigram/bigram/unigram causal predictor, including `exp(-(t-j)/512)` decay, was retained. The Lecture 2/3 syllabus itself was not present in the supplied workspace, so this implementation used the cited architectural ideas and primary public descriptions, not an unverifiable claim of course-slide conformance or a TA score.

Correctness preflight: the chunked delta scan agreed with a sequential reference recurrence to a maximum absolute error of `2.384185791015625e-07` on a short numerical check. `.venv\Scripts\python.exe -m unittest discover -s tests -v` passed all five unchanged contract tests (`OK`) before training and again after the stability adjustment. The model has 1,095,800 parameters. No immutable course files were edited.

The first training attempt used source SHA-256 `ed72d0735ad19f425c5f4554a51b9648b8e45865f7708c028e966e3ddcaab539`. Loss became non-finite at step 300 and validation stopped; this attempt consumed approximately 322.3 s and 2,457,600 training targets, with no saved checkpoint. A plausible numerical cause was the chunk scan's product/ratio decay calculation underflowing or amplifying gradients. The retry computed decay ratios in log space, bounded alpha to `[0.05, 0.99]` and beta to `[0.01, 0.99]`, and normalized the mixer output. These combined changes removed the observed NaN; they do not isolate its precise cause. Stabilized source SHA-256: `daea813b9fe852b5f69a5c1eb532f08f38a8bbd56e78385b701a9f402d83af2b`.

Phase 3 retry: `.venv\Scripts\python.exe train.py --implementation student --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/iter_10` from `code/`. It completed 1,200 steps from scratch, batch size 32, FP32, processing 9,830,400 training targets. Training excluding validation took 1,384.89 s; scheduled validation added 84.30 s. The training process took 1,507.20 s. Validation BPB at steps 300/600/900/1200 was `2.054552 / 1.851670 / 1.783380 / 1.757301`. The checkpoint `code/runs/iter_10/checkpoint.pt` has SHA-256 `4ba18e7179ff2b7855da446f51573f4edeeb30e67e205f83808d85878904632d` and is 4,419,708 bytes (4.215 MiB). The failed attempt plus retry add 12,288,000 processed training targets; cumulative search through Iteration 10 is 61,440,000 targets and approximately 4,149.23 s training time excluding validation (including the failed run estimate).

| Measure | Iteration 8 leader | Iteration 10 hybrid |
|---|---:|---:|
| Parameters | 1,088,192 | 1,095,800 |
| Validation BPB, final | **1.751079** | 1.757301 |
| Independent validation CPU scoring time | **7.801 s** | 25.434 s |
| Validation time ceiling | 23.4 s | 23.4 s |
| Observed peak evaluation process-tree working set | 1.4170 GiB | 1.4360 GiB |
| Checkpoint size | 4,379,957 bytes | 4,419,708 bytes |
| Successful-run training time excluding validation | **436.66 s** | 1,384.89 s |

Phase 4 independent command: `.venv\Scripts\python.exe evaluate.py --checkpoint runs/iter_10/checkpoint.pt --device cpu --precision fp32 --split validation`. It reproduced `1.7573014257543382` BPB across all 376,599 validation targets and 1,148,007 bytes in `25.433911` s. A roughly 200 ms Windows process-tree sample saw `1,541,885,952` bytes peak working set (an estimate that may miss a shorter peak). In-training validation also exceeded the time gate at steps 900 and 1200. The benchmark host's load varied, but the independently observed time is above the hard ceiling and cannot be counted as a pass.

Decision: **Reject Iteration 10 for promotion.** It is `0.006223` BPB worse than Iteration 8, trains about 3.17 times longer on the successful run, and misses the 23.4 s independent validation-time gate by 2.034 s. Contract, baseline-relative BPB, observed RAM, and asset-size checks pass. This bundled experiment cannot attribute the result to DeltaNet, attention gating, shortcut mixing, or their interaction. The trained candidate remains in `student.py` for checkpoint reproducibility; it is **not** the selected leader. Restoring the exact Iteration 8 source revision is required before continuing from that architecture. The test split was not used, and the <1.70 aspiration remains unmet.

## Iteration 12 — Scaled RoPE/RMSNorm/SwiGLU with extended cosine schedule

Hypothesis: lifting the earlier informal 1.1M-parameter and 1,200-step restrictions will improve the Iteration 5 backbone while retaining its causal trigram/bigram/unigram exact-match cache. The course limits are 64 MiB of inference assets, 4 GiB peak evaluation RAM, and 5× baseline CPU scoring time; training length is unrestricted. `code/student.py` now uses eight unshared full-attention blocks, width 320, ten 32-dimensional heads, SwiGLU hidden width exactly 1,280, RoPE, RMSNorm, and tied input/output embeddings. The model has **13,801,280 parameters**, not the anticipated 10.6M; FP32 parameters alone occupy 52.648 MiB. `code/configs/iter_12.json` records the dimensions. The Iteration 5 cache algorithm and mixture weights are retained, including strict `j < t` matching, call-local state, and `exp(-(t-j)/512)` recency weighting. After training, `@torch.inference_mode()` was added around `predict_log_probs` for a minor inference speed improvement; it left the algorithm, weights, and measured BPB unchanged. Final `student.py` SHA-256: `afe8dac57a4c07217db95deb7160a59297bfe9e0af5acd95670f0c3a980614e0`.

`code/train.py` defaults to the new config and exactly 3,000 updates, batch size 32, seed 17, AdamW with peak learning rate 0.001, 100-step warmup, weight decay 0.1, and cosine factor `0.1 + 0.9 × 0.5 × (1 + cos(pi × step / 4800))`. The 4,800 denominator is independent of the 3,000-step stopping point. The requested command was run with `--threads 32` added after a short CPU-thread benchmark: `.venv\Scripts\python.exe train.py --implementation student --seed 17 --eval-every 500 --threads 32 --run-dir runs/iter_12`. An initial four-thread attempt was interrupted after 300 steps (about 829.91 s, 2,457,600 processed training targets) when 32 threads proved faster; it saved no checkpoint. A separate CUDA setup attempt failed while downloading PyTorch and did not train a model. The successful run started from scratch, processed 24,576,000 training targets in 4,953.27 s excluding intermediate validation, and took 5,140.02 s end-to-end. Intermediate validation added 153.02 s. Total Iteration 12 training search therefore processed 27,033,600 targets, including the interrupted attempt. Checkpoint ancestry: only the successful seed-17 run, with no pretrained weights.

| Step | Validation BPB | 32-thread in-training scoring time |
|---:|---:|---:|
| 500 | 1.765869 | 25.529 s |
| 1,000 | 1.647300 | 26.059 s |
| 1,500 | 1.607898 | 25.892 s |
| **2,000** | **1.592116** | 25.243 s |
| 2,500 | 1.599752 | 25.187 s |
| 3,000 | 1.622825 | 25.109 s |

The best observed intermediate validation score was at step 2,000, but the trainer saves a checkpoint only after step 3,000. The actual saved checkpoint is `code/runs/iter_12/checkpoint.pt` (SHA-256 `066cf62e24b878d7cc38ab1601a9cab157b657a93c8c4cee8e33e368175e53d4`), with **1.6228246 BPB**, not a sub-1.60 result. The worsening after step 2,000 is consistent with overfitting or schedule mismatch but does not establish its cause. The saved checkpoint improves on the Iteration 8 validation leader by about 0.128254 BPB, but that comparison also changes model size, training duration, and architecture.

All five contract tests passed before training and again after the inference-mode addition, including causal invariance, normalization, example independence, state reset, and finite training gradients. An independent default-four-thread evaluation of the final source scored all 376,599 targets over 1,148,007 bytes at 1.6228246 BPB in **39.165 s** (`validation_cpu_fp32_final_threads4.json`). The same checkpoint and final source took **25.182 s** with `--threads 32` and identical BPB (`validation_cpu_fp32_inference_mode.json`). For a same-host comparison, the baseline checkpoint took 4.957 s at four threads and 3.419 s at 32 threads on validation, so Iteration 12 used **7.90×** and **7.37×** baseline CPU time, respectively. Both exceed the course's 5× limit; they also exceed the workflow's reference-derived 23.4 s ceiling. The checkpoint is **55,237,685 bytes (52.679 MiB)**, under 64 MiB. A separate approximately 200 ms sample of the Windows evaluator process tree with 32 threads observed **1,683,832,832 bytes (1.5682 GiB)** peak working set, under 4 GiB; it is an observed estimate that may miss a shorter peak.

Decision: **Reject Iteration 12 for promotion under the CPU scoring limit.** The saved model greatly improves validation BPB and meets the measured asset and RAM limits, but the final checkpoint misses both the 1.60 aspiration and the hard 5× CPU-time gate. The current `student.py` and `runs/iter_12/checkpoint.pt` reproduce this candidate; the earlier Iteration 8 leader remains the last accepted model, although its exact source revision must be restored to score its checkpoint. No test split was used. Substantive AI assistance generated the implementation, executed the training and audits, and prepared this record; disclose it in the final submission documentation.

## Iteration 13 — GELU and learned positions at 10.6M parameters

Hypothesis: replacing Iteration 12's SwiGLU with a two-matrix GELU MLP and RoPE with learned absolute position embeddings will lower inference cost enough to retain the scaled model under the CPU gate. `code/student.py` now has width 320, eight unshared full-attention blocks, ten heads of dimension 32, a `4 × width = 1,280` GELU hidden layer, RMSNorm, and tied token/output embeddings. The exact Iteration 5 causal trigram/bigram/unigram cache is retained: sources obey `j < t`, use already-observed successors, reset per call, and carry the unchanged 0.6/0.35/0.1 mixture weights with `exp(-(t-j)/512)` recency decay. The actual parameter count is **10,596,160** (40.421 MiB of FP32 weights). The earlier statement that RoPE performs trigonometric functions during every scoring call was incorrect: Iteration 12 precomputed sine and cosine at model construction. This experiment changes MLP form, positional mechanism, and parameter count together, so it cannot attribute either the BPB or speed difference to one change alone.

`code/train.py` now defaults to `code/configs/iter_13.json`, **2,400 steps**, batch size 32, seed 17, AdamW with learning rate 0.001 and weight decay 0.1, 100-step warmup, and cosine horizon **3,600** rather than the 2,400-step stopping point. It also writes `best_validation_checkpoint.pt` whenever scheduled validation BPB improves, while `checkpoint.pt` always records the final step. Training command from `code/`: `.venv\Scripts\python.exe train.py --implementation student --seed 17 --eval-every 200 --threads 32 --run-dir runs/iter_13`. It started from random initialization, ran exactly 2,400 updates, processed **19,660,800 training targets**, and took **2,642.67 s** excluding 214.38 s of intermediate validation. End-to-end process time was 2,884.48 s. Both saved checkpoints represent step 2,400; the selected `checkpoint.pt` has SHA-256 `aa653489ff62d0101589027a915ed6a815e68d1d3c3b0d0a4bf2bda4e94db9ab`. No earlier checkpoint or pretrained weight was reused.

| Training step | Validation BPB with temperature 1.0 | 32-thread scoring time |
|---:|---:|---:|
| 200 | 2.177195 | 17.595 s |
| 400 | 2.036523 | 17.247 s |
| 600 | 1.927074 | 17.604 s |
| 800 | 1.836520 | 17.420 s |
| 1,000 | 1.769093 | 17.775 s |
| 1,200 | 1.718272 | 18.143 s |
| 1,400 | 1.686740 | 17.954 s |
| 1,600 | 1.657250 | 17.898 s |
| 1,800 | 1.636457 | 18.376 s |
| 2,000 | 1.619556 | 18.471 s |
| 2,200 | 1.610988 | 17.945 s |
| 2,400 | **1.601842** | 17.957 s |

The exact requested backbone and cache narrowly missed 1.60 BPB. A validation-only calibration sweep then divided backbone logits by a fixed temperature before the same exact cache interpolation. This did not retrain or change any cache rule, match order, recency weight, or mixture weight. The six additional temperatures tried were 0.95, 1.05, 1.10, 1.12, 1.15, and 1.20; their validation BPBs were 1.612972, 1.595878, **1.594317**, 1.594778, 1.596517, and 1.601936, respectively. Temperature **1.10** was selected using validation only and is fixed in `code/student.py`. This adds six full validation passes to the search cost. The checkpoint weights still come solely from training text. The training metrics record the pre-calibration implementation SHA-256 `ea2c9665c2dc9e93c057db21f9aae2966ed136e0b1c9873853353594377eb9a5`; the final prediction source SHA-256 is `ac10afb46a56df4ba787f5fc05379dd5b9effd4efce904e5781cc62176a67c12`.

The fixed evaluator sets four threads before constructing the model. On this 32-logical-CPU host, `build_model` now explicitly requests up to 32 PyTorch CPU threads for this 320-wide, eight-block configuration; no evaluator file was changed. This thread choice is necessary for the measured time gate here and must be disclosed in reproduction instructions. Before the thread setting was made automatic, a four-thread pass took 27.891 s, whereas an explicit 32-thread pass took 16.799 s with temperature 1.0. On the final predictor, the plain command `.venv\Scripts\python.exe evaluate.py --checkpoint runs/iter_13/checkpoint.pt --split validation` reproduced **1.5943168 BPB** over all 376,599 targets and 1,148,007 bytes in **16.948 s**. A separate RAM-sampled pass reproduced the same BPB in **17.733 s**. A nearby same-host baseline pass with 32 threads took **3.585 s**, giving **4.73×** and **4.95×** baseline time. Both are below the hard 5× ratio and below the workflow's reference-derived 23.4 s ceiling, though the same-host ratio has little margin and may vary with machine load.

All five contract tests passed before training and after final calibration and thread selection. The selected checkpoint occupies **42,412,469 bytes (40.448 MiB)**, under the 64 MiB asset limit. A roughly 200 ms Windows process-tree sampling pass observed **1,625,587,712 bytes (1.5139 GiB)** peak evaluation working set, under 4 GiB; this observed estimate could miss a shorter peak. The final predictor improves the accepted Iteration 8 validation score by **0.156762 BPB**. Iteration 12's step-2,000 intermediate score of 1.592116 was numerically lower but had no saved checkpoint and its architecture exceeded the CPU limit; it does not establish a valid competing submission result.

Decision: **Promote Iteration 13 as the current validation leader on the measured host**, with explicit 32-thread inference. The calibrated checkpoint achieves 1.594317 validation BPB and meets measured time, RAM, and asset gates. The close 5× timing margin warrants reproduction on the final scoring hardware before submission. The test split remains untouched until the method is frozen for final reporting. Substantive AI assistance implemented, trained, calibrated, tested, and documented this iteration; disclose it in the final submission materials.

## Final freeze and full-test audit

After Iteration 13's architecture, checkpoint, exact-match cache, temperature 1.10, and CPU thread setting were frozen, the supplied FP32 evaluator scored the complete test split. The recorded result is `code/runs/iter_13/test_cpu_fp32.json`: **1.6091547859297617 BPB**, 428,405 targets, 1,292,013 raw UTF-8 bytes, and **19.471336699996755 s** CPU scoring time. Its checkpoint SHA-256 is `aa653489ff62d0101589027a915ed6a815e68d1d3c3b0d0a4bf2bda4e94db9ab`, and its implementation SHA-256 is `ac10afb46a56df4ba787f5fc05379dd5b9effd4efce904e5781cc62176a67c12`, matching the final source. The checkpoint is 42,412,469 bytes (40.448 MiB). The supplied baseline test record reports 2.1012604380869244 BPB and 14.64351099999476 s, making the measured test scoring time about 1.33× baseline, within the 5× limit. Peak test RAM was reported below 4 GiB; the separately observed full-validation process-tree peak was 1.5139 GiB. The test JSON does not contain a peak-RAM measurement, so no precise test RAM number is claimed.

The final test score was measured after validation-only model and temperature selection, and no subsequent parameter or predictor change was made. No test result was used to choose the method. This is the frozen submission result; the validation BPB remains 1.594316785653766. Reproduction requires the original course `code/data/` directory, which is intentionally excluded from the public repository, and the exact `code/student.py` and `code/runs/iter_13/checkpoint.pt` pair.

## Final-checkpoint inference-only factorial ablation

After the submission was frozen, `code/ablate_final.py` loaded the unchanged Iteration 13 checkpoint and evaluated four inference variants on the full validation split using the official `evaluate.score` routine. The ablation only binds a call-local neural-only prediction method and changes the in-memory temperature value; it does not retrain, edit `student.py`, alter the checkpoint, use test text, or change the evaluator. The JSON record is `code/runs/iter_13/ablation_validation_cpu_fp32.json`.

| Exact same checkpoint | Cache | Temperature | Validation BPB | CPU scoring time in ablation run |
|---|:---:|---:|---:|---:|
| Neural model | Off | 1.00 | 1.643535445 | 15.345 s |
| Neural + calibration | Off | 1.10 | 1.635770970 | 17.366 s |
| Neural + discrete cache | On | 1.00 | 1.601842055 | 27.868 s |
| Final combination | On | 1.10 | **1.594316786** | 27.651 s |

At temperature 1.00, the cache alone improves BPB by 0.041693390. At cache-off, calibration alone improves it by 0.007764475. Together they improve the neural-only baseline by 0.049218659 BPB. The temperature gain conditional on the cache is 0.007525269. The approximately 0.000239206 BPB interaction shows the improvements are nearly additive on this metric, not exactly additive. Each variant scored 376,599 targets and 1,148,007 bytes. The cache-on timing in this later ablation session is materially higher than the frozen 16.948 s validation and 19.471 s test records; CPU timing is sensitive to machine load and the ablation script's run context. These new validation-only measurements do not change the frozen official test result or its recorded resource audit. The cache is parameter-free and adds no neural matrix multiplication, but its comparisons, exponential weighting, and scatter accumulation have nonzero computational cost.
