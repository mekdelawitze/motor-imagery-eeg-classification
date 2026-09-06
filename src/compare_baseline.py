"""
Compare EEGNet vs the CSP+LDA classical baseline, subject by subject.
Reads results/accuracy_by_subject.csv (from run_all_subjects.py) and
results/baseline_accuracy.csv (from baseline.py --all), both of which must
already exist.

Usage:
    python -m src.compare_baseline
"""

import csv

import matplotlib.pyplot as plt


def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return {int(row["subject"]): float(row["val_accuracy"]) for row in reader}


def main():
    eegnet = read_csv("results/accuracy_by_subject.csv")
    baseline = read_csv("results/baseline_accuracy.csv")
    subjects = sorted(eegnet.keys())

    print(f"{'subject':>7} | {'EEGNet':>7} | {'CSP+LDA':>7} | {'diff':>7}")
    for s in subjects:
        diff = eegnet[s] - baseline[s]
        print(f"{s:>7} | {eegnet[s]:>7.3f} | {baseline[s]:>7.3f} | {diff:>+7.3f}")

    eegnet_mean = sum(eegnet.values()) / len(eegnet)
    baseline_mean = sum(baseline.values()) / len(baseline)
    eegnet_wins = sum(1 for s in subjects if eegnet[s] > baseline[s])
    print(f"\nmean: EEGNet {eegnet_mean:.3f}  CSP+LDA {baseline_mean:.3f}")
    print(f"EEGNet beats CSP+LDA on {eegnet_wins}/{len(subjects)} subjects")

    x = range(len(subjects))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], [eegnet[s] for s in subjects], width, label="EEGNet", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], [baseline[s] for s in subjects], width, label="CSP+LDA", color="#DD8452")
    ax.axhline(0.25, color="gray", linestyle="--", label="chance (25%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in subjects])
    ax.set_xlabel("subject")
    ax.set_ylabel("validation accuracy")
    ax.set_title("EEGNet vs classical CSP+LDA baseline, by subject")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/comparison_eegnet_vs_baseline.png", dpi=150)
    print("\nsaved results/comparison_eegnet_vs_baseline.png")


if __name__ == "__main__":
    main()
