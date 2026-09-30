"""
Classical baseline for comparison: CSP (Common Spatial Patterns) + LDA is the
standard classical benchmark for motor imagery EEG - the original EEGNet paper
itself compares against CSP+LDA-style methods. Same train/val split as the
deep learning model (session T train, session E val), so the comparison is
apples-to-apples.

By default this uses a fixed n_components=8, untuned - which isn't a fair
comparison against a model whose architecture was chosen deliberately. Use
--tune to instead pick CSP's n_components and LDA's shrinkage per subject via
cross-validation on the training session ONLY (session E / val is never
touched until the final score), matching the same discipline EEGNet's
training was held to: no peeking at validation data to pick settings.

Usage:
    python -m src.baseline --subject 3
    python -m src.baseline --all              # all 9 subjects, saves results/baseline_accuracy.csv
    python -m src.baseline --subject 3 --tune
    python -m src.baseline --all --tune        # saves results/baseline_accuracy_tuned.csv
"""

import argparse
import csv

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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


def tune_baseline(subject: int, n_components_grid=None, cv: int = 5, verbose: bool = True):
    """
    Cross-validated hyperparameter selection using ONLY the training session
    (session T). Session E (val) is untouched until the final .score() call,
    so this doesn't leak validation info into model selection. Note this is
    actually stricter than what EEGNet's own training does: EEGNet's
    checkpoint selection watches session E accuracy every epoch (see the
    README's methodology caveat), so this tuning procedure and EEGNet's
    training procedure are not held to the same rule.

    Grid: CSP n_components x {no LDA shrinkage, auto shrinkage}. Shrinkage
    needs the 'lsqr' solver (sklearn's default 'svd' solver doesn't support
    it), so the two options are swept as separate branches of the grid.
    """
    if n_components_grid is None:
        n_components_grid = [4, 6, 8, 10, 12]

    data = load_subject_data(subject)
    train_ds, val_ds = session_split(data)
    X_train, y_train = train_ds.X.numpy().astype(np.float64), train_ds.y.numpy()
    X_val, y_val = val_ds.X.numpy().astype(np.float64), val_ds.y.numpy()

    pipe = Pipeline([
        ("csp", CSP(reg=None, log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])
    param_grid = [
        {"csp__n_components": n_components_grid, "lda__solver": ["svd"]},
        {"csp__n_components": n_components_grid, "lda__solver": ["lsqr"], "lda__shrinkage": ["auto"]},
    ]
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    search = GridSearchCV(pipe, param_grid, cv=cv_splitter, scoring="accuracy", n_jobs=1)
    search.fit(X_train, y_train)

    val_acc = search.score(X_val, y_val)
    best = search.best_params_

    if verbose:
        print(
            f"subject {subject}: best params {best} "
            f"(cv accuracy on training session: {search.best_score_:.3f}) "
            f"-> val accuracy = {val_acc:.3f}"
        )
    return val_acc, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=None)
    p.add_argument("--all", action="store_true", help="run all 9 subjects")
    p.add_argument("--n-components", type=int, default=8, help="only used without --tune")
    p.add_argument("--tune", action="store_true", help="cross-validate n_components + LDA shrinkage on training session only")
    p.add_argument("--cv", type=int, default=5, help="CV folds for --tune")
    args = p.parse_args()

    if args.subject is None and not args.all:
        p.error("specify --subject N or --all")

    subjects = range(1, 10) if args.all else [args.subject]

    if args.tune:
        results = []
        for subject in subjects:
            val_acc, best = tune_baseline(subject, cv=args.cv)
            results.append((
                subject,
                val_acc,
                best.get("csp__n_components"),
                best.get("lda__solver"),
                best.get("lda__shrinkage", ""),
            ))
        if args.all:
            accs = [r[1] for r in results]
            mean_acc = sum(accs) / len(accs)
            print(f"\nmean tuned CSP+LDA accuracy across subjects: {mean_acc:.3f}")
            with open("results/baseline_accuracy_tuned.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["subject", "val_accuracy", "n_components", "lda_solver", "lda_shrinkage"])
                writer.writerows(results)
            print("saved results/baseline_accuracy_tuned.csv")
    elif args.all:
        results = []
        for subject in subjects:
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
    else:
        run_baseline(args.subject, args.n_components)


if __name__ == "__main__":
    main()
