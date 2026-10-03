"""
Step 3: evaluate on the TEST set (images the model has never seen).
Run:  python evaluate.py
Creates in outputs/: metrics.json, confusion_matrix.png, roc_curve.png, training_curves.png
"""
import json

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from tqdm import tqdm

import config
from src.data import get_loader
from src.model import load_model
from src.utils import get_device

CLASS_NAMES = ["real", "fake"]  # index 0 = real, 1 = fake


@torch.no_grad()
def predict_all(model, loader, device):
    probs, labels = [], []
    for x, y in tqdm(loader, desc="test"):
        logits = model(x.to(device)).squeeze(1)
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def plot_training_curves():
    path = config.OUTPUT_DIR / "history.json"
    if not path.exists():
        return
    h = json.loads(path.read_text())
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, h["train_loss"], "o-", label="train")
    axes[0].plot(epochs, h["val_loss"], "o-", label="validation")
    axes[0].set(title="Loss", xlabel="epoch")
    axes[1].plot(epochs, h["train_acc"], "o-", label="train")
    axes[1].plot(epochs, h["val_acc"], "o-", label="validation")
    axes[1].set(title="Accuracy", xlabel="epoch")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    device = get_device()
    model = load_model(config.CHECKPOINT_PATH, device)
    loader = get_loader("test")

    probs, labels = predict_all(model, loader, device)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision_fake": precision_score(labels, preds, zero_division=0),
        "recall_fake": recall_score(labels, preds, zero_division=0),
        "f1_fake": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan"),
        "n_test_images": int(len(labels)),
    }
    print("\n" + classification_report(labels, preds, target_names=CLASS_NAMES, digits=4, zero_division=0))
    for k, v in metrics.items():
        print(f"{k:>15}: {v:.4f}" if isinstance(v, float) else f"{k:>15}: {v}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr, label=f"CNN (AUC = {metrics['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="random guess")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(config.OUTPUT_DIR / "roc_curve.png", dpi=150)
        plt.close(fig)

    plot_training_curves()
    print(f"\nFigures and metrics saved in {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
