"""
Step 2: train the CNN.
Run:  python train.py
Saves the best model (by validation accuracy) to outputs/best_model.pt
and the loss/accuracy per epoch to outputs/history.json.
"""
import json
import time

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from src.data import get_loader
from src.model import build_model, save_checkpoint
from src.utils import get_device, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise just evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc="train" if training else "val  ", leave=False):
            x, y = x.to(device), y.float().to(device)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += ((logits > 0).float() == y).sum().item()
            n += x.size(0)

    return total_loss / n, correct / n


def main():
    set_seed(config.SEED)
    device = get_device()
    print(f"Device: {device} | model: {config.MODEL_NAME} | epochs: {config.EPOCHS}")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = get_loader("train")
    val_loader = get_loader("val")
    print(f"Train images: {len(train_loader.dataset)} | Val images: {len(val_loader.dataset)}")

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()  # binary classification loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        start = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc]):
            history[k].append(v)

        msg = (f"Epoch {epoch}/{config.EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
               f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - start:.0f}s")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_checkpoint(model, config.CHECKPOINT_PATH, epoch=epoch, val_acc=va_acc)
            msg += "  <- saved (best so far)"
        print(msg)

        with open(config.OUTPUT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {config.CHECKPOINT_PATH}")


if __name__ == "__main__":  # required on Windows when using DataLoader workers
    main()
