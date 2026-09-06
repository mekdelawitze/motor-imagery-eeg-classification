"""
Train EEGNet on BCI Competition IV Dataset 2a (one subject at a time).

Usage:
    python -m src.train --subject 1 --epochs 100 --batch-size 32
    python -m src.train --subject 1 --epochs 100 --plot   # also saves a training curve
"""

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import load_subject_data, session_split
from src.model import EEGNet
from src.utils import EarlyStopping, get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=1, help="BCI IV 2a subject id (1-9)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=30, help="early stopping patience (kept generous since this model trains fast, so premature stopping is a bigger risk than wasted time)")
    p.add_argument("--min-delta", type=float, default=0.005, help="minimum val loss improvement to reset early stopping patience")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true", help="save a training curve plot when done")
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total


def plot_training_curve(history: dict, subject: int) -> str:
    """Save a 2-panel plot (loss, accuracy) of train vs val over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(f"Subject {subject}: loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.axhline(0.25, color="gray", linestyle="--", label="chance")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title(f"Subject {subject}: accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()

    fig.tight_layout()
    out_path = f"results/training_curve_subject{subject}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def train_subject(
    subject: int,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 30,
    min_delta: float = 0.005,
    seed: int = 42,
    verbose: bool = True,
    plot: bool = False,
) -> float:
    """Train EEGNet on one subject. Returns the best validation accuracy reached."""
    set_seed(seed)
    device = get_device()

    if verbose:
        print(f"device: {device}")
        print(f"loading subject {subject} (first run downloads data to ~/mne_data)...")
    data = load_subject_data(subject)
    train_ds, val_ds = session_split(data)
    if verbose:
        print(f"train trials: {len(train_ds)}, val trials: {len(val_ds)}")
        print(f"classes: {data.label_names}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_channels, n_samples = data.X.shape[1], data.X.shape[2]
    model = EEGNet(n_classes=len(data.label_names), channels=n_channels, samples=n_samples).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    epoch_iter = tqdm(range(1, epochs + 1), desc=f"subject {subject}") if verbose else range(1, epochs + 1)
    for epoch in epoch_iter:
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"checkpoints/subject{subject}_best.pt")

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
                f"| val loss {val_loss:.4f} acc {val_acc:.3f}"
            )

        if stopper.step(val_loss):
            if verbose:
                print(f"early stopping at epoch {epoch}")
            break

    if verbose:
        print(f"\nbest validation accuracy for subject {subject}: {best_val_acc:.3f}")
        print(f"(chance level for {len(data.label_names)} classes: {1 / len(data.label_names):.3f})")

    if plot:
        out_path = plot_training_curve(history, subject)
        if verbose:
            print(f"saved {out_path}")

    return best_val_acc


def main():
    args = parse_args()
    train_subject(
        subject=args.subject,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        verbose=True,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
