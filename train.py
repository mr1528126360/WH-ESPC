#!/usr/bin/env python3
"""Train WH-ESPC on multi-sequence chest CT scans.

Example:
    python train.py \
        --csv data/labels.csv \
        --data-dir data/scans \
        --backbone-weights checkpoints/caformer_b36.sail_in22k.bin \
        --output-dir outputs
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from wh_espc import WHESPC, build_dataloaders
from wh_espc.engine import (evaluate, plot_confusion_matrix,
                            plot_roc_pr_curves, train_one_epoch)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train WH-ESPC on multi-sequence chest CT scans.")

    # Data
    parser.add_argument("--csv", default="data/labels.csv",
                        help="CSV file with one row per scan (folder name + label).")
    parser.add_argument("--data-dir", default="data/scans",
                        help="Root directory containing one folder of slices per scan.")
    parser.add_argument("--id-col", default="ID", help="CSV column with scan folder names.")
    parser.add_argument("--label-col", default="Label", help="CSV column with class labels.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of scans used for validation.")
    parser.add_argument("--seed", type=int, default=42)

    # Slice sampling along the axial Z-axis
    parser.add_argument("--begin-frame", type=int, default=4)
    parser.add_argument("--end-frame", type=int, default=40)
    parser.add_argument("--skip-frame", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)

    # Module 1: feature extraction
    parser.add_argument("--backbone", default="caformer_b36.sail_in22k",
                        help="timm model name of the frozen feature extractor.")
    parser.add_argument("--backbone-weights",
                        default="checkpoints/caformer_b36.sail_in22k.bin",
                        help="Local pre-trained backbone checkpoint; downloaded "
                             "from the Hugging Face Hub if not found.")
    parser.add_argument("--backbone-feature-dim", type=int, default=768,
                        help="Backbone output feature dimension.")
    parser.add_argument("--fc-hidden1", type=int, default=1024)
    parser.add_argument("--fc-hidden2", type=int, default=768)
    parser.add_argument("--embed-dim", type=int, default=512,
                        help="Latent embedding dimension shared by all modules.")
    parser.add_argument("--dropout", type=float, default=0.3)

    # Module 2: MoE transformer
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Step size of the dynamic bias adjuster.")
    parser.add_argument("--window-size", type=int, default=3,
                        help="Historical tracing window of the bias adjuster.")

    # Module 3: sequence classification
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-fc-dim", type=int, default=32)
    parser.add_argument("--bidirectional", action="store_true",
                        help="Use a Bi-LSTM decoder instead of a unidirectional LSTM.")
    parser.add_argument("--num-classes", type=int, default=3)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="e.g. 'cuda', 'cuda:0' or 'cpu'.")
    parser.add_argument("--output-dir", default="outputs")

    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader = build_dataloaders(
        csv_path=args.csv,
        data_dir=args.data_dir,
        id_col=args.id_col,
        label_col=args.label_col,
        begin_frame=args.begin_frame,
        end_frame=args.end_frame,
        skip_frame=args.skip_frame,
        image_size=args.image_size,
        test_size=args.test_size,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train scans: {len(train_loader.dataset)}, "
          f"validation scans: {len(val_loader.dataset)}")

    model = WHESPC(
        backbone_name=args.backbone,
        backbone_weights=args.backbone_weights,
        backbone_feature_dim=args.backbone_feature_dim,
        fc_hidden1=args.fc_hidden1,
        fc_hidden2=args.fc_hidden2,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        capacity_factor=args.capacity_factor,
        alpha=args.alpha,
        window_size=args.window_size,
        lstm_layers=args.lstm_layers,
        lstm_hidden=args.lstm_hidden,
        lstm_fc_dim=args.lstm_fc_dim,
        bidirectional=args.bidirectional,
        num_classes=args.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)

    best_f1 = -1.0
    best_epoch = -1
    best_checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
    history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.log_interval)
        metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{args.epochs} "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {metrics['loss']:.4f} acc {metrics['accuracy']:.4f} "
              f"macro-F1 {metrics['macro_f1']:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": metrics["loss"],
            "val_acc": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
        })

        # Save the checkpoint with the best validation macro-F1.
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch + 1
            torch.save({
                "epoch": best_epoch,
                "best_macro_f1": best_f1,
                "val_accuracy": metrics["accuracy"],
                "val_loss": metrics["loss"],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, best_checkpoint_path)
            print(f">>> Best macro-F1 updated: {best_f1:.4f} (epoch {best_epoch}), "
                  f"checkpoint saved to {best_checkpoint_path}")
            print(metrics["report"])

            # Evaluation figures of the best epoch.
            class_labels = list(range(args.num_classes))
            plot_confusion_matrix(metrics["y_true"], metrics["y_pred"],
                                  class_labels,
                                  os.path.join(args.output_dir, "confusion_matrix.png"))
            plot_roc_pr_curves(metrics["y_true"], metrics["y_score"],
                               args.num_classes,
                               os.path.join(args.output_dir, "roc_pr_curves.png"))

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"Best validation macro-F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
