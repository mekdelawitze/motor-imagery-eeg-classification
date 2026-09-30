"""
Check how much a subject's reported accuracy depends on the random seed, rather
than treating a single run's number as the "true" accuracy.

Usage:
    python -m src.seed_stability --subjects 7 --seeds 0,1,2,3,4
    python -m src.seed_stability --subjects 2,5,6,9 --seeds 0,1,2,3,4
"""

import argparse
import csv
import os
import statistics

from src.train import train_subject


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", type=str, required=True, help="comma-separated subject ids")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4", help="comma-separated seeds")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    subjects = [int(s) for s in args.subjects.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    all_results = {}
    for subject in subjects:
        accs = []
        for seed in seeds:
            acc = train_subject(
                subject=subject,
                epochs=args.epochs,
                patience=args.patience,
                seed=seed,
                verbose=False,
            )
            print(f"subject {subject} seed {seed}: {acc:.3f}")
            accs.append(acc)
        all_results[subject] = accs
        mean = statistics.mean(accs)
        median = statistics.median(accs)
        stdev = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(
            f"  -> subject {subject}: mean {mean:.3f}  median {median:.3f}  "
            f"stdev {stdev:.3f}  min {min(accs):.3f}  max {max(accs):.3f}\n"
        )

    print("\n==== summary ====")
    csv_path = "results/seed_stability.csv"
    # Merge with whatever's already on disk instead of overwriting it, so
    # running this on a subset of subjects (e.g. --subjects 7) doesn't erase
    # rows from an earlier run on a different subset (e.g. --subjects 2,5,6,9).
    # This is the fix for a real bug: this file used to open in "w" mode
    # unconditionally, which silently dropped subject 7's data when it was
    # run separately from the 2,5,6,9 run.
    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                existing_rows.append((int(row["subject"]), int(row["seed"]), float(row["val_accuracy"])))
    # Rows for subjects we just reran replace their old rows; everything else is kept.
    existing_rows = [r for r in existing_rows if r[0] not in all_results]
    new_rows = [
        (subject, seed, acc)
        for subject, accs in all_results.items()
        for seed, acc in zip(seeds, accs)
    ]
    merged_rows = sorted(existing_rows + new_rows, key=lambda r: (r[0], r[1]))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "seed", "val_accuracy"])
        for subject, seed, acc in merged_rows:
            writer.writerow([subject, seed, acc])
    for subject, accs in all_results.items():
        mean = statistics.mean(accs)
        median = statistics.median(accs)
        stdev = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"subject {subject}: mean {mean:.3f}  median {median:.3f}  stdev {stdev:.3f}")
    print("\nsaved results/seed_stability.csv")


if __name__ == "__main__":
    main()
