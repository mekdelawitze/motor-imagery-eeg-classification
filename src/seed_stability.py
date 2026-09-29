"""
Check how much a subject's reported accuracy depends on the random seed, rather
than treating a single run's number as the "true" accuracy.

Usage:
    python -m src.seed_stability --subjects 7 --seeds 0,1,2,3,4
    python -m src.seed_stability --subjects 2,5,6,9 --seeds 0,1,2,3,4
"""

import argparse
import csv
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
    with open("results/seed_stability.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "seed", "val_accuracy"])
        for subject, accs in all_results.items():
            for seed, acc in zip(seeds, accs):
                writer.writerow([subject, seed, acc])
    for subject, accs in all_results.items():
        mean = statistics.mean(accs)
        median = statistics.median(accs)
        stdev = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"subject {subject}: mean {mean:.3f}  median {median:.3f}  stdev {stdev:.3f}")
    print("\nsaved results/seed_stability.csv")


if __name__ == "__main__":
    main()
