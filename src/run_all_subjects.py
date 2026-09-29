"""
Train EEGNet independently on each of the 9 BCI IV 2a subjects with the same
hyperparameters, and report how much validation accuracy varies across people.

Usage:
    python -m src.run_all_subjects
"""

import csv

import matplotlib.pyplot as plt

from src.train import train_subject

SUBJECTS = list(range(1, 10))


def main():
    results = []
    for subject in SUBJECTS:
        print(f"\n=== subject {subject} ===")
        acc = train_subject(subject=subject, epochs=100, verbose=True)  # patience: use train_subject's default (currently 30)
        results.append((subject, acc))
        print(f"=== subject {subject} done: {acc:.3f} ===")

    accs = [acc for _, acc in results]
    mean_acc = sum(accs) / len(accs)
    min_acc = min(accs)
    max_acc = max(accs)

    print("\n\n==== summary across all subjects ====")
    for subject, acc in results:
        print(f"subject {subject}: {acc:.3f}")
    print(f"\nmean: {mean_acc:.3f}  min: {min_acc:.3f}  max: {max_acc:.3f}  range: {max_acc - min_acc:.3f}")

    with open("results/accuracy_by_subject.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "val_accuracy"])
        writer.writerows(results)
    print("\nsaved results/accuracy_by_subject.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    subjects, accs_list = zip(*results)
    ax.bar([str(s) for s in subjects], accs_list, color="#4C72B0")
    ax.axhline(0.25, color="gray", linestyle="--", label="chance (25%)")
    ax.axhline(mean_acc, color="darkorange", linestyle="-", label=f"mean ({mean_acc:.2f})")
    ax.set_xlabel("subject")
    ax.set_ylabel("validation accuracy")
    ax.set_title("EEGNet motor imagery accuracy by subject")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/accuracy_by_subject.png", dpi=150)
    print("saved results/accuracy_by_subject.png")


if __name__ == "__main__":
    main()
