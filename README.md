# WH-ESPC

Official implementation of **WH-ESPC**, a deep learning model for multi-sequence
chest CT classification.

## Model architecture

WH-ESPC consists of three key modules:

1. **Feature extraction module (CaFormer).** A pre-trained CaFormer
   (`caformer_b36.sail_in22k`) is employed as a frozen image feature extractor
   to sequentially extract abstract features from each CT slice, providing a
   solid foundation for the subsequent multi-sequence feature extraction
   module. A trainable fully-connected head projects the backbone features
   into a 512-dimensional latent embedding space.

2. **Multi-sequence CT feature extraction module (MoE with historical memory
   and load balancing).** A transformer whose feed-forward sub-layers are
   replaced by a sparse Mixture-of-Experts (MoE) network. The MoE uses a
   **dual-stage gated routing mechanism**: a noisy top-k router first produces
   routing probabilities and a per-expert load estimate; a dynamic bias
   adjuster then re-routes the tokens with bias-corrected logits. The
   **load-balancing weighting strategy** is based on historical information
   tracing (a sliding window over recent routing loads) and **softsign
   regularization**, which keeps the bias updates bounded. A shared expert
   processes every token. This module adaptively extracts critical information
   from multi-sequence CT images.

3. **Classification decision module (LSTM along the axial Z-axis).** A
   Bidirectional Long Short-Term Memory (Bi-LSTM) network processes the
   sequential features along the axial Z-axis (from lung apex to lung base),
   effectively capturing the spatial anatomical continuity across contiguous
   CT slices. The representation at the final time step is classified by a
   small fully-connected head.

```
CT scan (sequence of axial slices)
        │
        ▼
┌─────────────────────┐
│  CaFormerEncoder    │  frozen pre-trained CaFormer + trainable FC head
│  (per-slice embed)  │  [B, F, C, H, W] → [B, F, 512]
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  MoETransformer     │  2 × (multi-head self-attention + SparseMoE)
│  (dual-stage gated  │  noisy top-k routing → dynamic bias re-routing,
│   routing + load    │  historical-memory load tracing + softsign reg.,
│   balancing)        │  16 experts, top-4, 1 shared expert
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  LSTMDecoder        │  LSTM along the Z-axis + FC classifier
│  (apex → base)      │  [B, F, 512] → class probabilities
└─────────────────────┘
```

## Repository structure

```
WH-ESPC/
├── train.py                  # training / evaluation entry point
├── requirements.txt
├── wh_espc/
│   ├── __init__.py
│   ├── encoder.py            # Module 1: CaFormerEncoder
│   ├── moe.py                # Expert, NoisyTopkRouter, DynamicBiasAdjuster, SparseMoE
│   ├── transformer.py        # Module 2: MoETransformer
│   ├── decoder.py            # Module 3: LSTMDecoder
│   ├── model.py              # WHESPC full model
│   ├── dataset.py            # CTSequenceDataset + dataloader construction
│   └── engine.py             # train/eval loops and evaluation plots
└── checkpoints/              # pre-trained backbone weights (not tracked)
```

## Installation

```bash
pip install -r requirements.txt
```

Download the pre-trained CaFormer backbone weights
([`caformer_b36.sail_in22k`](https://huggingface.co/timm/caformer_b36.sail_in22k))
and place them at `checkpoints/caformer_b36.sail_in22k.bin`. If the file is
missing, the weights are downloaded automatically from the Hugging Face Hub
on first use.

## Data preparation

Each CT scan must be stored as a folder of axial slice images, sorted along
the Z-axis (lung apex → lung base):

```
data/
├── labels.csv          # one row per scan
└── scans/
    ├── <scan_id_1>/
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    ├── <scan_id_2>/
    └── ...
```

`labels.csv` must contain a folder-name column and an integer label column
(default column names `ID` and `Label`; configurable via `--id-col` /
`--label-col`):

```csv
ID,Label
scan_id_1,0
scan_id_2,2
```

By default, 18 slices per scan are sampled (slice indices 4, 6, …, 38 of the
naturally sorted slice list; see `--begin-frame/--end-frame/--skip-frame`).

## Training

```bash
python train.py \
    --csv data/labels.csv \
    --data-dir data/scans \
    --backbone-weights checkpoints/caformer_b36.sail_in22k.bin \
    --output-dir outputs
```

Default hyperparameters (matching the original experiments):

| Setting | Value |
|---|---|
| image size | 224 × 224 |
| slices per scan | 18 (indices 4–38, step 2) |
| latent embedding dim | 512 |
| MoE | 16 experts, top-4, 1 shared expert, capacity factor 1.0 |
| bias adjuster | α = 0.1, historical window T = 3, softsign gain 8 |
| transformer | 2 layers, 4 attention heads |
| LSTM decoder | 1 layer, 64 hidden units, FC dim 32 |
| optimizer | Adam, lr = 1e-5 |
| batch size / epochs | 64 / 100 |

The training split is class-balanced by random oversampling combined with
data augmentation (horizontal flip, rotation, color jitter, translation).
During training the checkpoint with the best validation macro-F1 is saved to
`outputs/best_model.pth`, together with the confusion matrix
(`confusion_matrix.png`), ROC/PR curves (`roc_pr_curves.png`) and the
training history (`history.json`).

Useful flags:

- `--bidirectional` — use the Bi-LSTM decoder variant.
- `--num-experts`, `--top-k`, `--alpha`, `--window-size` — MoE ablations.
- `--device cpu` — run without a GPU (slow).

Run `python train.py --help` for the full list of options.

## Notes on this implementation

The code was refactored from the original research notebook. Behavior is
faithful to the original experiments, with the following deliberate clean-ups:

- Slice files are sorted naturally before sampling, so the sequence order
  deterministically follows the axial Z-axis.
- The dynamic routing biases are updated only in training mode (not during
  evaluation).
- The three sub-networks are wrapped into a single `WHESPC` module with a
  single optimizer over all trainable parameters (the CaFormer backbone
  stays frozen).
- The train/validation split uses a fixed random seed for reproducibility.

## Citation

If you find this code useful, please cite our paper (citation to be added).
# WH-ESPCA: multi-encoder version

**WH-ESPCA** extends WH-ESPC from a single CaFormer encoder to three
heterogeneous frozen encoders with learnable gated late fusion:

1. **Multi-encoder feature extraction.** Three frozen ImageNet-pretrained
   backbones — CaFormer-B36 (768-d), ResNet-152 (2048-d) and Swin-B (1024-d) —
   extract per-slice embeddings in parallel. Each stream is projected into a
   shared 512-d latent space by its own trainable FC head.

2. **Per-stream MoE-Transformer.** Each stream is processed by an independent
   MoE-Transformer identical to WH-ESPC Module 2 (noisy top-k routing,
   dynamic bias adjustment with historical-memory load tracing and softsign
   regularisation, 16 experts, top-4, 1 shared expert, 2 layers, 4 heads).

3. **LSTM decoder + gated fusion.** Each stream is decoded by its own LSTM
   decoder (1 layer, 64 hidden units, FC dim 32) into class probabilities
   `p_m` and a 32-d stream hidden state `h_m`. A **sample-adaptive learnable
   gate** — a generalisation of USweA's fixed per-class accuracy weighting —
   computes per-sample stream weights
   `g = softmax(W [h_1; h_2; h_3])` and outputs the convex combination
   `p = Σ_m g_m · p_m`.

```
scan (slice sequence)
   ├── CaFormer-B36 ── FC head ── MoE-Transformer ── LSTM ── p1, h1 ──┐
   ├── ResNet-152  ── FC head ── MoE-Transformer ── LSTM ── p2, h2 ──┼─ gated fusion ── prediction
   └── Swin-B      ── FC head ── MoE-Transformer ── LSTM ── p3, h3 ──┘
```

## Files added by this update

```
wh_espc/
├── encoders.py          # FrozenBackboneEncoder registry + FCProjectionHead + feature extraction
├── moe_transformer.py   # MultiHeadAttention, SparseMoE, DynamicBiasAdjuster, MoETransformer
├── decoder_whespca.py   # LSTMDecoder (with hidden output) + GatedFusion
└── model_whespca.py     # WHESPCA full model
train_whespca.py         # training / evaluation entry point
```

## Training

```bash
python train_whespca.py \
    --csv data/labels.csv \
    --data-dir data/scans \
    --num-classes 4 \
    --runs 10 --output-dir outputs_whespca
```

Pretrained backbone weights are read from `checkpoints/` (see
`wh_espc/encoders.py::ENCODER_REGISTRY`); missing files fall back to
automatic download from the Hugging Face Hub.

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
over runs, including ECE) and `mean_roc/pr/dca.png`.
