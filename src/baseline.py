"""
Classical baseline for comparison: CSP (Common Spatial Patterns) + LDA is the
standard classical benchmark for motor imagery EEG - the original EEGNet paper
itself compares against CSP+LDA-style methods. Same train/val split as the
deep learning model (session T train, session E val), so the comparison is
apples-to-apples.

Usage:
    python -m src.baseline --subject 3
    python -m src.baseline --all       # all 9 subjects, saves results/baseline_accuracy.csv
"""

import argparse
import csv

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from src.dataset import load_subject_data, session_split


def run_baseline(subject: int, n_components: int = 8, verbose: bool = True) -> float:
    data = load_subject_data(subject)
    train_ds, val_ds = session_split(data)

    X_train, y_train = train_ds.X.numpy().astype(np.float64), train_ds.y.numpy()
    X_val, y_val = val_ds.X.numpy().astype(np.float64), val_ds.y.numpy()

    clf = Pipeline([
        ("csp", CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])
    clf.fit(X_train, y_train)
    acc = clf.score(X_val, y_val)

    if verbose:
        print(f"subject {subject}: CSP+LDA val accuracy = {acc:.3f}")
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=None)
    p.add_argument("--all", action="store_true", help="run all 9 subjects")
    p.add_argument("--n-components", type=int, default=8)
    args = p.parse_args()

    if args.all:
        results = []
        for subject in range(1, 10):
            acc = run_baseline(subject, args.n_components)
            results.append((subject, acc))
        accs = [a for _, a in results]
        mean_acc = sum(accs) / len(accs)
        print(f"\nmean CSP+LDA accuracy across subjects: {mean_acc:.3f}")
        with open("results/baseline_accuracy.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "val_accuracy"])
            writer.writerows(results)
        print("saved results/baseline_accuracy.csv")
    elif args.subject is not None:
        run_baseline(args.subject, args.n_components)
    else:
        p.error("specify --subject N or --all")


if __name__ == "__main__":
    main()
