# A Scaled Causal Decoder with Exact Within-Window Memory

**DASE7506 Mini Project 1 — Final technical report**
**Protocol:** `7506-mp1-wt2-v2` · **Frozen test BPB:** 1.609155 · **Checkpoint:** Iteration 13

## Abstract

We develop a language model for the supplied WikiText-2 benchmark in three phases: small-model architecture and memory experiments, capacity scaling under a CPU inference budget, and post-training calibration. The final predictor combines a 10,596,160-parameter causal decoder with a parameter-free exact-match cache over the current scoring window. A validation-selected softmax temperature of 1.10 completes the method. On the frozen full test split, it achieves **1.609155 bits per byte (BPB)**, versus **2.101260 BPB** for the supplied baseline. Recorded FP32 CPU scoring takes **19.471 s**, or **1.33×** the same-host baseline test time, and the **40.448 MiB** checkpoint fits the 64 MiB inference-asset limit. Reported peak RAM is below 4 GiB. A same-checkpoint 2×2 inference ablation shows that cache and calibration each improve validation BPB. The cache is free of trained parameters and added neural matrix multiplication, but it performs nonzero CPU computation.

## 1. Benchmark and experimental controls

The fixed course protocol uses a train-fitted BPE vocabulary of 2,048 tokens and independent causal windows of 256 positions. Every next-token target except the first token of the split is scored once. Let `L` be summed negative natural-log likelihood and `B` be the entire raw UTF-8 byte count. Then `BPB = L / (B ln 2)`. The validation split has 376,599 targets over 1,148,007 bytes; the test split has 428,405 targets over 1,292,013 bytes. All architecture, cache, checkpoint, and temperature choices used validation. We ran the complete test evaluator after freezing the final predictor.

The course limits uncompressed inference assets to 64 MiB, peak evaluation RAM to 4 GiB, and CPU scoring time to at most five times the baseline on a comparable host. Training time and model size are indirect constraints through those inference gates. The source, tokenizer, evaluator, and data are unchanged. The public repository omits raw course data; the original supplied `code/data/` bundle is required for reproduction and is verified by `common.py`.

## 2. Phase I — micro-scale architecture and memory (Iterations 1–11)

We first studied models around 1.1 million parameters under a matched 1,200-step, 9,830,400-target training budget. This small-scale setting made architectural comparisons and repeated inference tests tractable. The supplied 1,088,256-parameter baseline recorded 2.0710 validation BPB. Iteration 1 combined RoPE, RMSNorm, and SwiGLU in a 1,120,896-parameter decoder and reached 1.828568 BPB. Because all three changed together, that comparison establishes a combined gain, not a per-component cause. Iteration 2 kept the same seed and processed-target count but substituted learned absolute positions; it reached 2.006781 BPB with 1,153,664 parameters. The 0.178213 BPB RoPE advantage in this pair is specific to that small architecture; the learned position table also added parameters and changed initialization draws.

Iterations 3–7 explored shared blocks, position choices, and several cache forms. The important inference-only control is Iteration 5: the exact Iteration 1 weights scored **1.828568** without memory and **1.753179** with the hierarchical trigram/bigram/unigram cache. Thus the cache improved validation BPB by **0.075389** without retraining or changing parameter count. Four-token and confidence-gated variants did not surpass it. Iterations 8–11 tested further small-model alternatives, including multi-query attention and a hybrid recurrent mixer; the hybrid missed the CPU gate. Full variants, negative results, and search costs are in [EXPERIMENT.md](EXPERIMENT.md).

### 2.1 Discrete exact-match causal cache

At query position `t`, the cache searches prior positions `j < t` for the longest available matching suffix: trigram, then bigram, then unigram. A selected source contributes its already-observed successor `x_{j+1}` with weight `w(t,j) = exp(-(t-j)/512)`. If `M_t` is the set of selected sources, its normalized vote distribution is

`q_t(v) = [Σ_{j∈M_t} w(t,j) 1{x_{j+1}=v}] / [Σ_{j∈M_t} w(t,j)]`.

The mixture share `α_t` is 0.60, 0.35, or 0.10 for trigram, bigram, or unigram matches, respectively, and zero with no match. Every source satisfies `j < t`, so `j+1 ≤ t`: the voted successor is inside the observed prefix. All tensors are local to the current call, with no cross-window state. The cache adds no trained parameters and no neural matrix multiplication. It is **not literally zero-FLOP**: equality comparisons, recency exponentials, accumulation, and normalization consume CPU time and memory. The provided contract suite verifies causal invariance, output normalization, independent examples, state reset, and finite training gradients.

## 3. Phase II — capacity scaling and the CPU bottleneck (Iterations 12–13)

Iteration 12 scaled to eight 320-wide blocks while retaining SwiGLU and RoPE. The approximately 13.8-million-parameter model reached an intermediate **1.592116 validation BPB at step 2,000**, but that intermediate state was not checkpointed. Its saved step-3,000 checkpoint scored **1.622825 BPB** and required **39.165 s** for four-thread validation, or **25.182 s** with 32 threads; both exceeded the corresponding five-times-baseline comparisons. Its checkpoint fit the asset budget, but CPU time prevented promotion.

Iteration 13 made an **architectural diet** for the scaled decoder: a two-linear-layer **4× GELU MLP** replaced SwiGLU, and learned absolute position embeddings replaced RoPE. The final architecture has eight unshared self-attention blocks, width 320, ten heads of dimension 32, MLP width 1,280, RMSNorm, tied input/output embeddings, and **10,596,160 parameters**. On the same 32-thread validation setting, the earlier saved candidate took **25.182 s** and the final calibrated predictor took **16.948 s**, a 32.7% lower recorded time. The final complete-test run took **19.471 s**. Comparing the earlier 39.165-second validation run directly with the final 19.471-second *test* run would mix thread settings and splits. The position mechanism, MLP, parameter count, and thread policy changed together, so these experiments do not isolate SwiGLU or RoPE as individually responsible for a specific number of seconds. Earlier RoPE sine/cosine tables were precomputed at model construction, not recalculated per token.

The final model was trained **from random initialization** for 2,400 updates at batch size 32 and seed 17, processing **19,660,800 training targets**. AdamW used weight decay 0.1, 100 warmup steps, peak learning rate `10^-3`, minimum `10^-4`, and a cosine horizon of 3,600 steps beyond the stopping point. Recorded training time was **2,642.67 s** excluding 214.38 s of scheduled validation, and end-to-end process time was 2,884.48 s on the 32-thread CPU host. No earlier checkpoint or pretrained weights were inherited. The uncalibrated cache-enabled final checkpoint scored **1.601842 validation BPB**.

## 4. Phase III — post-training temperature calibration

With trained weights and cache fixed, we divided backbone logits by a temperature `T` before mixing with the cache:

`p_t(v) = (1-α_t) softmax(z_t/T)[v] + α_t q_t(v)`.

Validation-only trials at `T = 0.95, 1.05, 1.10, 1.12, 1.15, 1.20` yielded BPB `1.612972, 1.595878, 1.594317, 1.594778, 1.596517, 1.601936`, respectively. We froze **T = 1.10** before full-test evaluation. This changes no weights, cache matches, or scoring protocol.

### 4.1 Same-checkpoint inference-only ablation

To separate cache and calibration effects in the final model, [ablate_final.py](code/ablate_final.py) loaded the **same frozen Iteration 13 checkpoint** for a 2×2 validation study. It used `evaluate.score`, changed only the in-memory temperature, and replaced `predict_log_probs` with the neural softmax when the cache was disabled. Neither training nor a test evaluation was performed for these variants.

| Final checkpoint variant | Cache | Temperature | Validation BPB | Gain versus neural T=1.0 |
|---|:---:|---:|---:|---:|
| Neural decoder | Off | 1.00 | 1.643535445 | — |
| Decoder + calibration | Off | 1.10 | 1.635770970 | 0.007764475 |
| Decoder + exact cache | On | 1.00 | 1.601842055 | 0.041693390 |
| **Full predictor** | **On** | **1.10** | **1.594316786** | **0.049218659** |

With the cache enabled, calibration contributes **0.007525269 BPB**. The two gains are nearly additive; their interaction on BPB is about **0.000239206**. The [ablation JSON](code/runs/iter_13/ablation_validation_cpu_fp32.json) records full target counts, bytes, hashes, and per-run times. Its later cache-enabled passes took about 27.7 s on the host, materially slower than the frozen 16.948 s validation and 19.471 s test records. CPU timing depends on load and run context; the new ablation timings do not replace the official test audit.

## 5. Frozen result and resource audit

| Measure | Baseline test | Final frozen test | Limit |
|---|---:|---:|---:|
| BPB | 2.1012604 | **1.6091548** | Minimize |
| FP32 CPU scoring time | 14.6435 s | **19.4713 s (1.33×)** | ≤5× baseline = 73.2176 s |
| Uncompressed checkpoint | — | **42,412,469 bytes (40.448 MiB)** | ≤64 MiB inference assets |
| Peak evaluation RAM | — | **<4 GiB reported** | ≤4 GiB |

The final test improvement over baseline is **0.492106 BPB**, or 23.42% relative. The result is recorded in [test_cpu_fp32.json](code/runs/iter_13/test_cpu_fp32.json). A separate approximately 200-ms-sampled full-validation pass observed **1.5139 GiB** peak evaluator process-tree working set. The test JSON has no peak-RAM field, so no precise test peak is asserted; sampling can also miss shorter spikes. On same-host 32-thread validation, the final predictor took 16.948 s against a 3.585 s baseline (4.73×); another final pass took 17.733 s (4.95×). This narrower validation margin makes cross-host timing verification important despite the comfortable recorded test ratio.

The exact final checkpoint SHA-256 is `aa653489ff62d0101589027a915ed6a815e68d1d3c3b0d0a4bf2bda4e94db9ab`; the matching predictor source SHA-256 is `ac10afb46a56df4ba787f5fc05379dd5b9effd4efce904e5781cc62176a67c12`. The repository [README](README.md) gives installation, test, training, and direct evaluation commands.

## 6. Limitations, attribution, and disclosure

The cache performs quadratic comparisons within fixed 256-token windows. The architectural diet is a grouped intervention rather than a per-layer or per-operation profile. Validation informed 13 iterations and a temperature sweep, so its final score has selection bias; the full test result was obtained after the predictor was frozen. Timing and RAM estimates depend on the host and measurement method. No external training text, pretrained weights, cached test answers, or cross-window state were used.

The WikiText-2 benchmark was introduced by Merity and colleagues and contains text by Wikipedia contributors. The upstream dataset notices identify CC BY-SA 3.0 and GNU FDL obligations; raw data are excluded from this public repository. Substantive AI assistance contributed implementation, experiment execution, audits, analysis, and report drafting. The submitter is responsible for reviewing the code and reported claims.

## References

1. Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2016). [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843).
2. Zhang, B. and Sennrich, R. (2019). [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).
3. Hendrycks, D. and Gimpel, K. (2016). [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415).
4. Su, J. et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
