#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WH-ESPCA training / evaluation entry point (multi-encoder version).

Data convention (same as the single-encoder WH-ESPC repo):
  data/
  ├── labels.csv          # one row per scan (default columns: ID, Label)
  └── scans/<scan_id>/*.jpg

Example:
  python train_whespca.py --csv data/labels.csv --data-dir data/scans \
      --num-classes 4 --runs 10 --output-dir outputs_whespca
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

from wh_espc.encoders import ENCODER_REGISTRY, extract_features
from wh_espc.model_whespca import WHESPCA

GRID_FPR = np.linspace(0, 1, 101)
GRID_REC = np.linspace(0, 1, 101)
GRID_TH = np.linspace(0.01, 0.99, 99)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_scans(csv_path, data_dir, id_col, label_col):
    df = pd.read_csv(csv_path, dtype=str)
    label_map = {r[id_col].strip(): int(r[label_col]) for _, r in df.iterrows()}
    scans = []
    for sid in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, sid)
        if sid not in label_map or not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d)
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")))
        if files:
            scans.append((sid, label_map[sid], [os.path.join(d, f) for f in files]))
    labels = np.array([s[1] for s in scans])
    log(f"数据: {len(scans)} 例, 标签分布 {np.bincount(labels).tolist()}")
    return scans, labels


def oversample_to_balance(idx, labels, rng, cap=None):
    idx = np.asarray(idx)
    classes, counts = np.unique(labels[idx], return_counts=True)
    max_n = counts.max() if cap is None else min(counts.max(), cap)
    return np.concatenate([rng.choice(idx[labels[idx] == c], size=max_n, replace=True)
                           for c in classes])


def predict(model, batch_fn, n, bs, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, n, bs):
            idx = torch.arange(i, min(i + bs, n), device=device)
            probs.append(model(batch_fn(idx)).float().cpu().numpy())
    return np.concatenate(probs, axis=0)


def fit(model, batch_tr, ytr_t, batch_va, yva, lr, epochs, bs, device, seed, log_every=10):
    """Train; keep the checkpoint with the best validation macro-F1."""
    if seed is not None:
        set_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_tr = ytr_t.shape[0]
    best_f1, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        ep_loss = 0.0
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(batch_tr(idx)), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        probs = predict(model, batch_va, len(yva), bs, device)
        f1 = f1_score(yva, probs.argmax(1), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % log_every == 0 or ep == 0:
            log(f"  epoch {ep + 1}/{epochs} loss={ep_loss / n_tr:.4f} valF1={f1:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return predict(model, batch_va, len(yva), bs, device)


def compute_ece(y_true, probs, n_bins=15):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def dca_curve(yb, prob, thresholds):
    n = len(yb)
    return np.array([(((prob >= pt) & (yb == 1)).sum() - ((prob >= pt) & (yb == 0)).sum()
                      * pt / (1 - pt)) / n for pt in thresholds])


def evaluate_and_save(run_idx, y_true, probs, run_dir, seed, num_classes):
    os.makedirs(run_dir, exist_ok=True)
    K = num_classes
    y_pred = probs.argmax(1)
    metrics = {
        "run": run_idx, "seed": seed,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc": (roc_auc_score(y_true, probs[:, 1]) if K == 2 else
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro")),
        "ece": compute_ece(y_true, probs),
    }
    with open(os.path.join(run_dir, "classification_report.txt"), "w") as f:
        f.write(f"# WH-ESPCA run={run_idx} seed={seed}\n\n")
        f.write(classification_report(y_true, y_pred, digits=4, zero_division=0))
        f.write("\nconfusion matrix:\n" + np.array2string(confusion_matrix(y_true, y_pred)))
        f.write("\n\nmetrics: " + json.dumps({k: round(v, 4) for k, v in metrics.items()
                                            if isinstance(v, float)}) + "\n")
    pred_df = {"y_true": y_true, "y_pred": y_pred}
    for c in range(K):
        pred_df[f"prob_{c}"] = probs[:, c]
    pd.DataFrame(pred_df).to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

    tpr_m, prec_m, nb_m = [], [], []
    for c in range(K):
        yb = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(yb, probs[:, c])
        tpr_m.append(np.interp(GRID_FPR, fpr, tpr))
        prec, rec, _ = precision_recall_curve(yb, probs[:, c])
        prec_m.append(np.interp(GRID_REC, rec[::-1], prec[::-1]))
        nb_m.append(dca_curve(yb, probs[:, c], GRID_TH))
    tpr_m, prec_m, nb_m = np.array(tpr_m), np.array(prec_m), np.array(nb_m)

    for arr, grid, xl, yl, name in [
            (tpr_m, GRID_FPR, "False Positive Rate", "True Positive Rate", "roc"),
            (prec_m, GRID_REC, "Recall", "Precision", "pr"),
            (nb_m, GRID_TH, "Threshold probability", "Net benefit", "dca")]:
        fig, ax = plt.subplots(figsize=(5, 5))
        for c in range(K):
            ax.plot(grid, arr[c], lw=1, alpha=0.6, label=f"class_{c}")
        ax.plot(grid, arr.mean(0), lw=2.5, color="black", label="macro")
        if name == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=1)
        if name == "dca":
            ax.axhline(0, color="k", lw=1, ls=":")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"WH-ESPCA {name.upper()} (run {run_idx})"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(run_dir, f"{name}.png"), dpi=150)
        plt.close(fig)

    np.savez(os.path.join(run_dir, "curves.npz"), tpr=tpr_m, prec=prec_m, nb=nb_m,
             macro_tpr=tpr_m.mean(0), macro_prec=prec_m.mean(0), macro_nb=nb_m.mean(0))
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def aggregate(out_dir, runs):
    rows = []
    for r in range(runs):
        p = os.path.join(out_dir, "WH-ESPCA", f"run_{r}", "metrics.json")
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "all_runs_metrics.csv"), index=False)
    summ = {"method": "WH-ESPCA", "n_runs": len(df)}
    for c in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc", "ece"]:
        if c in df and df[c].notna().any():
            summ[f"{c}_mean"], summ[f"{c}_sd"] = df[c].mean(), df[c].std(ddof=1)
    pd.DataFrame([summ]).to_csv(os.path.join(out_dir, "summary_mean_sd.csv"), index=False)
    log("mean ± SD: " + "  ".join(f"{c}={summ[f'{c}_mean']:.4f}±{summ[f'{c}_sd']:.4f}"
                                  for c in ["accuracy", "f1_macro", "auc", "ece"]
                                  if f"{c}_mean" in summ))
    tprs, precs, nbs = [], [], []
    for r in range(runs):
        p = os.path.join(out_dir, "WH-ESPCA", f"run_{r}", "curves.npz")
        if os.path.exists(p):
            d = np.load(p)
            tprs.append(d["macro_tpr"]); precs.append(d["macro_prec"]); nbs.append(d["macro_nb"])
    for arr, grid, xl, yl, name in [
            (np.array(tprs), GRID_FPR, "False Positive Rate", "True Positive Rate", "roc"),
            (np.array(precs), GRID_REC, "Recall", "Precision", "pr"),
            (np.array(nbs), GRID_TH, "Threshold probability", "Net benefit", "dca")]:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(grid, arr.mean(0))
        ax.fill_between(grid, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), alpha=0.2)
        if name == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=1)
        if name == "dca":
            ax.axhline(0, color="k", lw=1, ls=":")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"WH-ESPCA mean {name.upper()} ({len(arr)} runs)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "WH-ESPCA", f"mean_{name}.png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="WH-ESPCA (multi-encoder WH-ESPC)")
    ap.add_argument("--csv", required=True, help="labels.csv (ID, Label)")
    ap.add_argument("--data-dir", required=True, help="scans directory")
    ap.add_argument("--id-col", default="ID")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--encoders", default="caformer,resnet152,swin",
                    help="逗号分隔的编码器列表, 可选: " + ",".join(ENCODER_REGISTRY))
    ap.add_argument("--n-frames", type=int, default=8,
                    help="每例抽取帧数, 不足用末帧填充")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--cache-dir", default="feature_cache")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--oversample-cap", type=int, default=None)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--random-seed", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", default="outputs_whespca")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    enc_keys = [k.strip() for k in args.encoders.split(",")]
    for k in enc_keys:
        assert k in ENCODER_REGISTRY, f"未知编码器 {k}, 可选: {list(ENCODER_REGISTRY)}"

    scans, labels = load_scans(args.csv, args.data_dir, args.id_col, args.label_col)

    # 一次性提取各路特征 (多次运行共享, 磁盘缓存)
    feats = {}
    for k in enc_keys:
        feats[k] = extract_features(
            k, scans, args.n_frames, args.device,
            weights_path=os.path.join(args.checkpoint_dir, ENCODER_REGISTRY[k][1]),
            cache_path=os.path.join(args.cache_dir, f"feat__{k}.npy"))
    in_dims = [feats[k].shape[-1] for k in enc_keys]

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    for run in range(args.runs):
        seed = None if args.random_seed else args.base_seed + run
        rng = np.random if args.random_seed else np.random.RandomState(seed)
        tr_clean, te = train_test_split(np.arange(len(scans)), test_size=args.test_size,
                                        stratify=labels, random_state=seed, shuffle=True)
        tr = oversample_to_balance(tr_clean, labels, rng, cap=args.oversample_cap)

        def make_batch_fn(idxs):
            Xs = [torch.from_numpy(feats[k][idxs].astype(np.float32)).to(args.device)
                  for k in enc_keys]

            def fn(idx):
                return [X[idx] for X in Xs]
            return fn

        run_dir = os.path.join(args.output_dir, "WH-ESPCA", f"run_{run}")
        if os.path.exists(os.path.join(run_dir, "metrics.json")) and not args.force:
            log(f"跳过 run {run} (已存在结果, --force 可强制重跑)")
            continue
        log(f"##### run {run + 1}/{args.runs} seed={seed} "
            f"train={len(tr_clean)}(过采样后{len(tr)}) test={len(te)} #####")
        model = WHESPCA(in_dims, args.num_classes, drop_p=args.dropout)
        ytr_t = torch.from_numpy(labels[tr]).long().to(args.device)
        probs = fit(model, make_batch_fn(tr), ytr_t, make_batch_fn(te), labels[te],
                    lr=args.lr, epochs=args.epochs, bs=args.batch_size,
                    device=args.device, seed=seed)
        metrics = evaluate_and_save(run, labels[te], probs, run_dir, seed, args.num_classes)
        log(f"  run {run}: acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} "
            f"auc={metrics['auc']:.4f} ece={metrics['ece']:.4f}")
        del model
        torch.cuda.empty_cache()

    aggregate(args.output_dir, args.runs)
    log(f"全部完成, 总耗时 {(time.time() - t0) / 3600:.2f} 小时, 结果目录: {args.output_dir}")


if __name__ == "__main__":
    main()
