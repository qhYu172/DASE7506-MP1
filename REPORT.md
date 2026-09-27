# R-Drop Training and Checkpoint Averaging for a CPU-Bounded Causal Decoder

**Yu Qinghao · Student ID 3036483599**

**DASE7506 Mini Project 1 · Protocol `7506-mp1-wt2-v2`**

## Abstract

This work develops a compact language model under three hard inference limits: at most 64 MiB of assets, 4 GiB peak RAM, and five times the baseline CPU scoring time. The final predictor combines a 10,596,160-parameter causal Transformer, a call-local exact-match cache, temperature calibration, 19,200-step R-Drop training, and uniform averaging of four late checkpoints. It reaches **1.475055 validation BPB** and **1.493095 test BPB**, improving the prior Iteration 13 test result by 0.116060 BPB. On the paired validation audit it takes 29.492 s versus 6.799 s for the baseline (**4.34×**). The 42.946 MiB checkpoint and a sampled 1.5508 GiB process-tree working set also pass their gates. Checkpoint averaging contributes 0.005348 validation BPB over the raw final update. The proposed 1.30–1.40 range was not reached.

## 1. Task and controls

The fixed benchmark uses a train-fitted 2,048-token BPE vocabulary and independent causal windows of 256 tokens. Every target after the first token is scored once. For summed negative log likelihood \(L\) and the complete raw UTF-8 byte count \(B\),

\[
\mathrm{BPB}=\frac{L}{B\ln 2}.
\]

Validation contains 376,599 targets and 1,148,007 bytes; test contains 428,405 targets and 1,292,013 bytes. Architecture and hyperparameter choices used validation. The final test evaluation was run after the Iteration 14 method and averaged weights were fixed. Raw course data are omitted from the repository.

## 2. Development path

### 2.1 Phase I: micro-scale architecture and memory

Iterations 1–11 deliberately used models near 1.1M parameters and a matched 1,200-step budget to compare architecture and memory ideas cheaply. RoPE, RMSNorm, and SwiGLU jointly reduced validation BPB from the 2.0710 baseline to 1.828568. An inference-only control then applied the exact-match cache to unchanged weights and reduced BPB from 1.828568 to 1.753179. This 0.075389 gain established the cache independently of training.

### 2.2 Phase II: scaling and the architectural diet

Iteration 12 scaled to eight 320-wide blocks while retaining RoPE and SwiGLU. Its saved checkpoint did not satisfy the paired CPU gate. Iteration 13 replaced SwiGLU with a two-linear-layer 4× GELU MLP and RoPE with learned absolute positions. The resulting inference backbone has eight unshared blocks, width 320, ten 32-dimensional heads, MLP width 1,280, RMSNorm, tied input/output embeddings, and 10,596,160 parameters. This grouped **architectural diet** reduced parameter count and CPU work; the experiment does not isolate the individual timing contribution of either replacement.

### 2.3 Phase III: cache and calibration

At position \(t\), the cache finds strictly earlier positions \(j<t\) with the longest available matching suffix: trigram, then bigram, then unigram. A match votes for its observed successor \(x_{j+1}\) with recency weight

\[
w(t,j)=\exp(-(t-j)/512), \qquad
q_t(v)=\frac{\sum_{j\in M_t}w(t,j)\mathbf{1}[x_{j+1}=v]}
{\sum_{j\in M_t}w(t,j)}.
\]

The cache share is 0.60, 0.35, or 0.10 for trigram, bigram, or unigram matches. Because \(j+1\le t\), every voted token is already observed. State is recreated for every call, so information cannot cross examples or windows. The cache has no trainable parameters or neural matrix multiplications, although comparisons, exponentials, and accumulation require CPU work.

Backbone logits use the validation-selected temperature \(T=1.10\):

\[
p_t(v)=(1-\alpha_t)\operatorname{softmax}(z_t/T)_v+\alpha_tq_t(v).
\]

On the Iteration 13 checkpoint, the neural model at \(T=1.0\) scored 1.643535 validation BPB; calibration alone scored 1.635771, cache alone scored 1.601842, and both scored 1.594317. This is an inference-only 2×2 ablation with identical weights.

## 3. Iteration 14: R-Drop and checkpoint averaging

The final run started from random initialization and processed 157,286,400 training targets in 19,200 updates at batch size 32. AdamW used weight decay 0.1, 100 warmup updates, and cosine decay from \(10^{-3}\) to \(10^{-4}\) through update 19,200. Training took 3,150.09 s on an NVIDIA RTX 4060 Laptop GPU using BF16.

R-Drop requires stochastic views, so dropout 0.1 was added to attention and MLP residual branches during training only. For two forward distributions \(p_1,p_2\) and targets \(y\), the optimized loss was

\[
\mathcal L=\tfrac12[\mathrm{CE}(y,p_1)+\mathrm{CE}(y,p_2)]
+\tfrac12[\mathrm{KL}(p_1\Vert p_2)+\mathrm{KL}(p_2\Vert p_1)].
\]

The KL coefficient is 1.0. Evaluation bypasses dropout, retaining a deterministic inference graph and the same parameter count. Checkpoints were saved at updates 17,760, 18,240, 18,720, and 19,200. `average_checkpoints.py` uniformly averages every floating-point parameter in FP32. The raw update-19,200 checkpoint scored 1.480403 validation BPB; averaging improved this to **1.475055**, a gain of 0.005348.

## 4. Results and resource audit

| Measure | Iteration 13 | Iteration 14 final | Gate / reference |
|---|---:|---:|---:|
| Validation BPB | 1.594317 | **1.475055** | lower is better |
| Test BPB | 1.609155 | **1.493095** | lower is better |
| Parameters | 10,596,160 | **10,596,160** | — |
| FP32 CPU validation time | — | **29.492 s** | baseline 6.799 s; **4.34× ≤ 5×** |
| FP32 CPU test time | 19.471 s | **34.212 s** | complete test run |
| Checkpoint | 40.448 MiB | **42.946 MiB** | ≤64 MiB |
| Sampled evaluation RAM | — | **1.5508 GiB** | ≤4 GiB |

The final test score improves on Iteration 13 by 0.116060 BPB. The validation timing comparison uses the same host, PyTorch build, thread count, and split. CPU timings varied with host load, so the ratio is more meaningful than comparison with an earlier absolute time. RAM was sampled from the evaluator process tree at roughly 200 ms intervals and may miss shorter spikes.

The final checkpoint SHA-256 is `6703409e2705750f546d18c10465c0fbc26c268fa749c9d43900dc49192e155f`; the predictor source SHA-256 is `5d2b65057a2a9a22e9009d575444cb3659074a163e1386e92d722a6c7e50d213`.

## 5. Interpretation and limitations

Long training plus R-Drop yields the largest improvement in the project, while late checkpoint averaging adds a smaller measured gain. Because training duration, dropout, and R-Drop changed together relative to Iteration 13, their individual effects are not identified. Only raw-update versus averaged-checkpoint performance is isolated within Iteration 14. The exact-match cache performs quadratic comparisons inside each fixed 256-token window. Validation guided repeated development and therefore has selection bias. No external training text, teacher model, pretrained weights, test-answer cache, or cross-window state was used.

The peer PDF named in the task was not present in the workspace, so its reported 1.456 BPB and ablation values could not be independently checked and are not used as measured results here. The achieved score is above 1.456 and above the proposed 1.30–1.40 target.

## 6. Reproduction

From `code/`, with the original course `data/` directory installed:

```bash
python -m unittest discover -s tests -v
python evaluate.py --checkpoint runs/iter_14/checkpoint.pt --device cpu --precision fp32 --threads 32 --split validation
python evaluate.py --checkpoint runs/iter_14/checkpoint.pt --device cpu --precision fp32 --threads 32 --split test
```

To reproduce training and averaging:

```bash
python train.py --implementation student --config configs/iter_14.json --device cuda --precision bf16 --threads 4 --seed 17 --steps 19200 --cosine-horizon 19200 --batch-size 32 --rdrop-alpha 1.0 --save-steps 17760,18240,18720,19200 --run-dir runs/iter_14_reproduction
python average_checkpoints.py --run-dir runs/iter_14_reproduction
```

## References

1. Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). *Pointer Sentinel Mixture Models*. arXiv:1609.07843.
2. Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS.
3. Hendrycks, D., & Gimpel, K. (2016). *Gaussian Error Linear Units*. arXiv:1606.08415.
4. Liang, X. et al. (2021). *R-Drop: Regularized Dropout for Neural Networks*. NeurIPS.
5. Izmailov, P. et al. (2018). *Averaging Weights Leads to Wider Optima and Better Generalization*. UAI.

WikiText-2 contains text from Wikipedia contributors. Raw data are excluded from the repository. Substantive AI assistance contributed implementation, experiment execution, audits, analysis, and drafting; the author reviewed the final materials.
