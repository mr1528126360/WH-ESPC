# WH-ESPC / WH-ESPCA

Official implementation of **WH-ESPCA** (multi-encoder version) and the
original **WH-ESPC** (single-encoder version) — deep learning models for
multi-sequence medical image classification.

---

# WH-ESPCA: multi-encoder version (main)

**WH-ESPCA** uses three heterogeneous frozen encoders with per-stream
dynamic-bias MoE-Transformers and learnable gated late fusion:

1. **Multi-encoder feature extraction.** Three frozen ImageNet-pretrained
   backbones — CaFormer-B36 (768-d), ResNet-152 (2048-d) and Swin-B (1024-d) —
   extract per-slice embeddings in parallel. Each stream is projected into a
   shared 512-d latent space by its own trainable FC head.

2. **Per-stream MoE-Transformer.** Each stream is processed by an independent
   MoE-Transformer (noisy top-k routing, dynamic bias adjustment with
   historical-memory load tracing and softsign regularisation, 16 experts,
   top-4, 1 shared expert, 2 layers, 4 heads).

3. **LSTM decoder + gated fusion.** Each stream is decoded by its own LSTM
   decoder (1 layer, 64 hidden units, FC dim 32) into class probabilities
   `p_m` and a 32-d stream hidden state `h_m`. A **sample-adaptive learnable
   gate** — a learnable generalisation of USweA's fixed per-class accuracy
   weighting — computes per-sample stream weights
   `g = softmax(W [h_1; h_2; h_3])` and outputs the convex combination
   `p = Σ_m g_m · p_m`.

```
scan (slice sequence)
   ├── CaFormer-B36 ── FC head ── MoE-Transformer ── LSTM ── p1, h1 ──┐
   ├── ResNet-152  ── FC head ── MoE-Transformer ── LSTM ── p2, h2 ──┼─ gated fusion ── prediction
   └── Swin-B      ── FC head ── MoE-Transformer ── LSTM ── p3, h3 ──┘
```

## Repository structure

```
WH-ESPC/
├── train.py                  # WH-ESPC (single-encoder) entry point
├── train_whespca.py          # WH-ESPCA (multi-encoder) entry point
├── requirements.txt
├── wh_espc/
│   ├── __init__.py
│   ├── encoder.py            # WH-ESPC Module 1: CaFormerEncoder
│   ├── moe.py                # WH-ESPC: Expert, NoisyTopkRouter, DynamicBiasAdjuster, SparseMoE
│   ├── transformer.py        # WH-ESPC Module 2: MoETransformer
│   ├── decoder.py            # WH-ESPC Module 3: LSTMDecoder
│   ├── model.py              # WH-ESPC full model
│   ├── dataset.py            # CTSequenceDataset + dataloader construction
│   ├── engine.py             # train/eval loops and evaluation plots
│   ├── encoders.py           # WH-ESPCA: encoder registry + FCProjectionHead + feature extraction
│   ├── moe_transformer.py    # WH-ESPCA: MultiHeadAttention, SparseMoE, DynamicBiasAdjuster
│   ├── decoder_whespca.py    # WH-ESPCA: LSTMDecoder (with hidden output) + GatedFusion
│   └── model_whespca.py      # WH-ESPCA full model
└── checkpoints/              # pre-trained backbone weights (not tracked)
```

## Installation

```bash
pip install -r requirements.txt
```

Pretrained backbone weights are read from `checkpoints/` (see
`wh_espc/encoders.py::ENCODER_REGISTRY`); missing files are downloaded
automatically from the Hugging Face Hub on first use.

## Data preparation

Each scan must be stored as a folder of slice images:

```
data/
├── labels.csv          # one row per scan (default columns: ID, Label)
└── scans/
    ├── <scan_id_1>/
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    └── <scan_id_2>/
```

## Training (WH-ESPCA)

```bash
python train_whespca.py \
    --csv data/labels.csv \
    --data-dir data/scans \
    --num-classes 4 \
    --runs 10 --output-dir outputs_whespca
```

Default hyperparameters (matching the reported experiments):

| Setting | Value |
| --- | --- |
| encoders | CaFormer-B36 + ResNet-152 + Swin-B (all frozen) |
| slices per scan | 8 (evenly spaced, last-slice padding) |
| latent dim / MoE | 512 / 16 experts, top-4, 1 shared expert |
| fusion | learnable sample-adaptive gate over stream hidden states |
| optimizer | Adam, lr = 1e-5 |
| dropout / epochs / batch size | 0.5 / 60 / 64 |

Outputs per run: `classification_report.txt`, `roc.png`, `pr.png`, `dca.png`,
`predictions.csv`, `metrics.json`; aggregated `summary_mean_sd.csv` (mean ± SD
over runs, including ECE) and `mean_roc/pr/dca.png`. Useful flags:
`--encoders` (choose encoder subset), `--n-frames`, `--runs`, `--force`
(re-run ignoring cached results). Run `python train_whespca.py --help` for the
full list.

---

# WH-ESPC: single-encoder version (brief)

The original WH-ESPC shares the same MoE-Transformer and LSTM decoder, but
uses a **single frozen CaFormer-B36 encoder** instead of three encoders with
gated fusion:

```
CT scan ── CaFormerEncoder (frozen + FC head, 512-d)
       ── MoETransformer (noisy top-k + dynamic bias, 16 experts, top-4)
       ── LSTMDecoder (Z-axis) ── class probabilities
```

Train it with:

```bash
python train.py \
    --csv data/labels.csv \
    --data-dir data/scans \
    --backbone-weights checkpoints/caformer_b36.sail_in22k.bin \
    --output-dir outputs
```

Key settings: 224 × 224 input, 18 slices per scan (indices 4–38, step 2),
Adam lr = 1e-5, batch size 64, 100 epochs; best validation macro-F1
checkpoint is saved with confusion matrix and ROC/PR curves. Useful flags:
`--bidirectional`, `--num-experts`, `--top-k`, `--alpha`, `--window-size`.
Run `python train.py --help` for the full list.

## Citation

If you find this code useful, please cite our paper (citation to be added).
