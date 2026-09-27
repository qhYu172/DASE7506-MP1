# DASE7506 Mini Project 1 — Final Language Model Submission

**Yu Qinghao · Student ID 3036483599**

The final Iteration 14 predictor achieves **1.4930947 test BPB** and **1.4750551 validation BPB**. The matching [checkpoint](code/runs/iter_14/checkpoint.pt) is included, so evaluation does not require retraining. See the [PDF report](REPORT.pdf), [Markdown report](REPORT.md), [LaTeX source](REPORT.tex), and complete [experiment log](EXPERIMENT.md).

| Result or resource | Iteration 14 | Limit / reference |
|---|---:|---:|
| Full-test BPB | **1.4930947** | Iteration 13: 1.6091548 |
| Validation BPB | **1.4750551** | Iteration 13: 1.5943168 |
| FP32 CPU validation time | **29.492 s** | Same-host baseline: 6.799 s; ratio **4.34× ≤ 5×** |
| FP32 CPU test time | **34.212 s** | Recorded complete-test run |
| Peak evaluation RAM | **1.5508 GiB sampled** | ≤4 GiB |
| Checkpoint size | **45,031,925 bytes (42.946 MiB)** | ≤64 MiB |

CPU time varies with the host and system load. The compliance ratio above uses validation runs measured on the same host with 32 PyTorch threads. The roughly 200 ms RAM sampler can miss a shorter spike.

## Final method

The inference model keeps the efficient Iteration 13 architecture: eight causal attention blocks, width 320, ten heads, 4× GELU MLPs, learned absolute positions, RMSNorm, and tied embeddings. It has **10,596,160 parameters**.

Iteration 14 trained this model from scratch for **19,200 updates** with R-Drop. Training-only dropout 0.1 creates two stochastic views of each batch; the loss averages their cross-entropies and adds symmetric KL divergence with coefficient 1.0. Dropout is bypassed at evaluation, leaving the deterministic inference graph unchanged. AdamW uses a 100-step warmup and cosine decay from `0.001` to `0.0001` over all 19,200 steps.

After training, [average_checkpoints.py](code/average_checkpoints.py) uniformly averages FP32 weights from steps 17,760, 18,240, 18,720, and 19,200. Averaging improves validation BPB from 1.4804027 for the raw final checkpoint to **1.4750551**.

The predictor retains the call-local discrete exact-match cache. At each position it selects the longest available trigram, bigram, or unigram match from strictly earlier positions, weights observed successors by `exp(-(t-j)/512)`, and mixes the result with backbone probabilities. Its shares are 0.60, 0.35, and 0.10. A fixed validation-selected temperature of **1.10** is applied to backbone logits. The cache adds no learned parameters or neural matrix multiplications, but its comparisons and accumulation require real CPU work.

## Reproduce evaluation

Use Python 3.12 and the original course dataset and tokenizer. Raw data are intentionally excluded. Place the supplied course `data/` directory at `code/data/`; `common.py` verifies its manifest and hashes.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
cd code
python -m unittest discover -s tests -v
python evaluate.py --checkpoint runs/iter_14/checkpoint.pt --device cpu --precision fp32 --split validation
python evaluate.py --checkpoint runs/iter_14/checkpoint.pt --device cpu --precision fp32 --split test
```

The recorded outputs are [validation_cpu_fp32_final.json](code/runs/iter_14/validation_cpu_fp32_final.json) and [test_cpu_fp32.json](code/runs/iter_14/test_cpu_fp32.json).

## Reproduce training and averaging

The recorded run used PyTorch 2.7.1+cu126 in BF16 on an RTX 4060 Laptop GPU and took 3,150 seconds excluding final validation. From `code/`:

```bash
python train.py --implementation student --config configs/iter_14.json --device cuda --precision bf16 --threads 4 --seed 17 --steps 19200 --cosine-horizon 19200 --batch-size 32 --rdrop-alpha 1.0 --save-steps 17760,18240,18720,19200 --run-dir runs/iter_14_reproduction
python average_checkpoints.py --run-dir runs/iter_14_reproduction
python evaluate.py --checkpoint runs/iter_14_reproduction/checkpoint.pt --device cpu --precision fp32 --split validation
```

The run processed 157,286,400 training targets. Training from scratch may vary across hardware and software versions; the committed checkpoint is the final evaluation artifact.

Verify the frozen files:

```bash
sha256sum code/runs/iter_14/checkpoint.pt code/student.py
```

Expected SHA-256 values are `6703409e2705750f546d18c10465c0fbc26c268fa749c9d43900dc49192e155f` for the checkpoint and `5d2b65057a2a9a22e9009d575444cb3659074a163e1386e92d722a6c7e50d213` for the predictor source.

WikiText-2 was introduced by [Merity et al.](https://arxiv.org/abs/1609.07843); its text is from Wikipedia contributors. Substantive AI assistance contributed implementation, experiment execution, audits, analysis, and documentation. The submitter reviewed and is responsible for the final submission.
