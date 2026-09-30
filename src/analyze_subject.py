"""
Load a trained subject's checkpoint and inspect where it goes wrong: prints a
per-class breakdown and saves a confusion matrix plot.

Loads checkpoints/subject{N}_seed{seed}_best.pt (seed 42 by default, matching
train_subject()'s default). If you've since run a seed-stability sweep for
this subject, re-run `python -m src.train --subject N --epochs 100` first to
regenerate the seed-42 checkpoint - otherwise there may not be a matching
file on disk (checkpoints are seed-tagged specifically so a seed sweep can't
silently swap out the model this script analyzes).

Usage:
    python -m src.analyze_subject --subject 6
    python -m src.analyze_subject --subject 6 --seed 0   # analyze a specific seed's model
"""

import argparse

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import load_subject_data, session_split
from src.model import EEGNet
from src.utils import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--seed", type=int, default=42, help="which seed's checkpoint to load (default: 42, the canonical run)")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    data = load_subject_data(args.subject)
    _, val_ds = session_split(data)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    n_channels, n_samples = data.X.shape[1], data.X.shape[2]
    model = EEGNet(n_classes=len(data.label_names), channels=n_channels, samples=n_samples).to(device)

    ckpt_path = f"checkpoints/subject{args.subject}_seed{args.seed}_best.pt"
    print(f"loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            preds = model(X).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    print(f"\n=== subject {args.subject} ===")
    print(classification_report(all_labels, all_preds, target_names=data.label_names, digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    print("confusion matrix (rows = true label, columns = predicted label):")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.label_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Subject {args.subject} confusion matrix (seed {args.seed})")
    fig.tight_layout()
    out_path = f"results/confusion_subject{args.subject}_seed{args.seed}.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
