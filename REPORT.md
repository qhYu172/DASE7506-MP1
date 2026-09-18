# A Scaled Causal Decoder with Exact Within-Window Memory for WikiText-2

**DASE7506 Mini Project 1 — Final report**
**Protocol:** `7506-mp1-wt2-v2` · **Frozen test BPB:** 1.609155 · **Checkpoint:** Iteration 13

## Abstract

We train a 10,596,160-parameter decoder from scratch on the supplied WikiText-2 training split and combine it with a parameter-free, within-window exact-match cache. A validation-calibrated temperature of 1.10 sharpens the final comparison between the decoder and cache. The frozen predictor scores the complete test split at **1.609155 bits per byte (BPB)**, compared with **2.101260 BPB** for the supplied baseline. Its recorded FP32 CPU scoring time is **19.471 s**, checkpoint size **40.448 MiB**, and reported peak RAM is below **4 GiB**. The method fits the 5× CPU, 4 GiB RAM, and 64 MiB asset constraints. The final decoder uses a computationally cheaper GELU MLP and learned absolute positions after an earlier SwiGLU/RoPE candidate exceeded the CPU time gate. We document controlled comparisons, the non-isolated parts of the final architecture change, and the validation-only selection process.

## 1. Task and evaluation protocol

The course scorer uses a fixed train-fitted BPE tokenizer with 2,048 tokens and independent causal windows of up to 256 target tokens. Every target except the first token of each split is scored once; state cannot pass between windows. If `L` is total next-token negative log likelihood in natural units and `B` is the raw split byte length, then `BPB = L / (B ln 2)`. Lower is better. The full validation split has 376,599 targets and 1,148,007 UTF-8 bytes; the full test split has 428,405 targets and 1,292,013 bytes. All development decisions used validation. The final test run followed the freeze of architecture, weights, cache, temperature, and CPU thread setting.

The supplied baseline is a 1,088,256-parameter GPT-like model. Its recorded full-test BPB is 2.101260. The final test improvement is 0.492106 BPB, or 23.42% relative. We report complete-split FP32 CPU results, not token perplexity or a short sample.

## 2. Method

### 2.1 Scaled decoder and architectural diet

The final decoder has **eight unshared causal self-attention blocks**, width **320**, **ten 32-dimensional heads**, RMSNorm, and a **1,280-unit GELU MLP** in each block. Its 256-position learned embedding table replaces rotary positional embeddings (RoPE); input and output token embeddings are tied. Total trainable parameters are **10,596,160**. The approximately 40.421 MiB of FP32 parameter values, together with checkpoint metadata, yield a **42,412,469-byte (40.448 MiB)** checkpoint.

This layout resulted from an architectural diet. Iteration 12 used eight 320-wide blocks with SwiGLU and RoPE and had 13.8 million parameters. Its saved step-3,000 checkpoint scored 1.622825 validation BPB, but took **39.165 s** on four CPU threads and **25.182 s** on 32 threads, failing the corresponding 5× baseline CPU comparisons. The final GELU/absolute-position predictor took **16.948 s** on the same 32-thread validation setting and **19.471 s** on the full test split. The 32-thread validation comparison represents a 32.7% time reduction. The often-cited 39.165-to-19.471-second comparison mixes thread settings and data splits, so it is not a controlled speedup estimate. Parameter count, MLP, positions, and thread policy changed together. Also, the earlier RoPE implementation precomputed trigonometric tables at construction; repeated trigonometric evaluation was not its measured bottleneck.

### 2.2 Discrete exact-match causal cache

The decoder output is supplemented with a call-local cache of previously observed successors. At target position `t`, a source position `j` is eligible only if **`j < t`**. Sources are selected by the longest available matching suffix, first the current trigram, then bigram, then unigram. Each source votes for the already-observed next token `ids[j+1]`, with recency weight `exp(-(t-j)/512)`. Votes are normalized into a cache distribution `q_t`. Because `j < t`, every voted successor satisfies `j+1 <= t`; it is within the observed prefix. Masks are strictly lower triangular, examples are independent, and no state survives a call or crosses scoring windows.

For the selected order, the cache share is `alpha = 0.60` (trigram), `0.35` (bigram), or `0.10` (unigram); without a match, `alpha = 0`. The final probability is

`p_t(v) = (1 - alpha) softmax(logits_t / 1.10)[v] + alpha q_t(v)`.

The cache has **zero trainable parameters** and adds **no neural matrix multiplication**. It is not literally zero-FLOP: comparison, exponential weighting, scatter accumulation, and normalization consume CPU time and memory. The five supplied contract tests cover causal invariance, normalized finite probabilities, example independence, call reset, and training gradients.

### 2.3 Post-training calibration

The final checkpoint's backbone, before temperature calibration, scored **1.601842 validation BPB**. Holding the checkpoint and cache fixed, we evaluated temperatures `0.95`, `1.05`, `1.10`, `1.12`, `1.15`, and `1.20` on validation only. Their BPBs were `1.612972`, `1.595878`, **`1.594317`**, `1.594778`, `1.596517`, and `1.601936`; thus 1.10 was frozen. This calibration changes probabilities at inference without retraining or reading test text.

## 3. Training and comparisons

The final model began from random initialization. AdamW used batch size 32, seed 17, peak learning rate `1e-3`, minimum `1e-4`, weight decay 0.1, and 100 warmup steps. The cosine schedule has a **3,600-step horizon**, while training stops at **2,400 steps**, processing **19,660,800 training targets**. Reported training time was **2,642.67 s** excluding 214.38 s of scheduled validation; the whole process took **2,884.48 s** on the recorded 32-thread CPU host. The validation curve improved through step 2,400. Temperature selection added six complete validation passes. The [experiment log](EXPERIMENT.md) records earlier unsuccessful runs and search costs; this run did not inherit any checkpoint.

| Comparison | Parameters | Processed training targets | Validation BPB | Interpretation |
|---|---:|---:|---:|---|
| Course baseline | 1,088,256 | 9,830,400 | 2.0710 | Supplied reference |
| Iteration 1: RoPE/RMSNorm/SwiGLU, no cache | 1,120,896 | 9,830,400 | 1.828568 | Same-target comparison with baseline; multiple features changed |
| Iteration 2: learned positions ablation | 1,153,664 | 9,830,400 | 2.006781 | Same seed/budget as Iteration 1; position table also adds parameters |
| Iteration 5: Iteration 1 checkpoint + exact cache | 1,120,896 | 9,830,400 | 1.753179 | Same weights as Iteration 1; cache improves BPB by 0.075389 |
| Iteration 12: scaled SwiGLU/RoPE | ~13.8M | 24,576,000 | 1.622825 | Saved checkpoint; fails CPU time limit |
| Iteration 13: final calibrated model | **10,596,160** | **19,660,800** | **1.594317** | Selected on validation; passes recorded resource limits |

The Iteration 1 versus Iteration 2 pair is a focused position-mechanism ablation at the same seed and training-target budget. In that smaller setting, RoPE helped BPB by 0.178213; this does not imply it would help under the final, larger model's CPU constraint. The strongest cache ablation reuses the exact Iteration 1 checkpoint: adding the cache alone reduces validation BPB from 1.828568 to 1.753179. Iteration 13's gain over earlier small models also includes more capacity and training, so it cannot be assigned to a single architectural choice. Iteration 12 briefly reached 1.592116 validation BPB at step 2,000, but that intermediate state was not checkpointed; its saved model scored 1.622825 and exceeded the CPU gate.

## 4. Frozen result and resource audit

| Measure | Baseline test | Final test | Requirement |
|---|---:|---:|---:|
| BPB | 2.1012604 | **1.6091548** | Minimize |
| FP32 CPU scoring time | 14.6435 s | **19.4713 s** | At most 5× baseline = 73.2176 s |
| CPU-time ratio | 1.00× | **1.33×** | ≤5× |
| Checkpoint size | — | **40.448 MiB** | ≤64 MiB assets |
| Peak RAM | — | **<4 GiB reported** | ≤4 GiB |

The final test result is stored in [test_cpu_fp32.json](code/runs/iter_13/test_cpu_fp32.json). A separate validation run sampled 1.5139 GiB peak process-tree working set at roughly 200 ms intervals. The test JSON does not record peak RAM, and that validation sample can miss a shorter spike; no exact test RAM number is asserted. A 32-thread same-host baseline validation run took 3.585 s, versus 16.948 s for the final predictor (4.73×). A second final validation pass took 17.733 s (4.95× that baseline), showing that the validation timing margin can be narrow even though the recorded test baseline ratio is comfortable. Cross-machine timing must be measured anew.

The final checkpoint SHA-256 is `aa653489ff62d0101589027a915ed6a815e68d1d3c3b0d0a4bf2bda4e94db9ab`; the matching `student.py` SHA-256 is `ac10afb46a56df4ba787f5fc05379dd5b9effd4efce904e5781cc62176a67c12`. Installation, contract tests, training, and direct evaluation commands are in the [repository README](README.md). Raw course data are excluded from Git; reproducibility requires placing the original unmodified course `data/` bundle in `code/data/`. The provided `common.py` checks its manifest and hashes.

## 5. Limitations, data, and AI disclosure

The exact-match cache can add quadratic within-window comparison work, although the window length is fixed at 256. The final architecture comparison changes several factors at once. Validation was used repeatedly across 13 iterations, so its score is selection-biased; the full test evaluation was performed after the method was frozen. The measured result relies on the recorded CPU threading and software environment. Peak test RAM lacks a precise in-JSON measurement.

Only supplied training text was used to learn weights. The tokenizer, data, fixed evaluator, and official test split were not modified. WikiText-2 is attributed to Merity and colleagues and Wikipedia contributors; the dataset's upstream notices identify CC BY-SA 3.0 and GNU FDL obligations. The public repository omits raw dataset files. Substantive AI assistance contributed implementation, experimentation, auditing, analysis, and report drafting; the submitter is responsible for review and the final submission.

## References

1. Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2016). [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). Introduces WikiText.
2. Zhang, B. and Sennrich, R. (2019). [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).
3. Hendrycks, D. and Gimpel, K. (2016). [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415).
4. Su, J. et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
