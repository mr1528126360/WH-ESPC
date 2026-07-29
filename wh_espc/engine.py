"""Training and evaluation loops plus evaluation plots for WH-ESPC."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_curve, roc_curve)
from sklearn.preprocessing import label_binarize


def train_one_epoch(model, train_loader, optimizer, device, epoch,
                    log_interval=10):
    """Train the model for one epoch.

    Returns:
        (average loss, average accuracy) over the epoch.
    """
    model.train()
    losses, scores = [], []
    n_count = 0  # total number of trained samples in this epoch

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device).view(-1)
        n_count += x.size(0)

        optimizer.zero_grad()
        output, _ = model(x)

        # NOTE: the decoder returns softmax probabilities; feeding them to
        # cross_entropy (instead of raw logits) reproduces the original
        # training procedure.
        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        y_pred = torch.max(output, 1)[1]
        step_score = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
        scores.append(step_score)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Acc: {:.2f}%'.format(
                epoch + 1, n_count, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                loss.item(), 100 * step_score))

    return float(np.mean(losses)), float(np.mean(scores))


def evaluate(model, val_loader, device):
    """Evaluate the model.

    Returns:
        dict with keys ``loss``, ``accuracy``, ``macro_f1``, ``report``,
        ``y_true``, ``y_pred`` and ``y_score`` (class probabilities).
    """
    model.eval()
    test_loss = 0.0
    all_y, all_y_pred, all_y_score = [], [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device).view(-1)
            output, _ = model(x)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()

            y_pred = output.max(1, keepdim=True)[1]
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            all_y_score.extend(output)

    test_loss /= len(val_loader.dataset)
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    y_true = all_y.cpu().numpy().squeeze()
    y_pred = all_y_pred.cpu().numpy().squeeze()
    y_score = torch.stack(all_y_score).cpu().numpy()

    return {
        "loss": test_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "report": classification_report(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def plot_confusion_matrix(y_true, y_pred, class_labels, save_path):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_labels)),
           yticks=np.arange(len(class_labels)),
           xticklabels=class_labels, yticklabels=class_labels,
           xlabel='Predicted Label', ylabel='True Label',
           title='Confusion Matrix')
    threshold = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > threshold else 'black')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return cm


def plot_roc_pr_curves(y_true, y_score, num_classes, save_path):
    """Plot and save one-vs-rest ROC and precision-recall curves."""
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curves.
    ax = axes[0]
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'Class {i} ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
           xlabel='False positive rate', ylabel='True positive rate',
           title='ROC')
    ax.legend(loc="lower right")

    # Precision-recall curves.
    ax = axes[1]
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i],
                                                      y_score[:, i])
        ax.plot(recall, precision, color=colors[i], lw=2,
                label=f'Class {i} PR')
    ax.set(xlabel='Recall', ylabel='Precision', title='PR')
    ax.legend(loc='lower left')

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
