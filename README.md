# Motor Imagery EEG Classification with Deep Learning

Classifying imagined limb movement (left hand, right hand, feet, tongue) from raw EEG using an
EEGNet-style CNN in PyTorch, compared against a classical CSP+LDA baseline.

Dataset: BCI Competition IV Dataset 2a (9 subjects, 22 EEG channels, 4-class motor imagery),
loaded via [MOABB](https://neurotechx.github.io/moabb/) (MOABB names it `BNCI2014_001`).

## Key results

- EEGNet: 60.9% mean validation accuracy across 9 subjects (chance = 25%)
- CSP+LDA baseline: 60.3% mean validation accuracy - EEGNet is slightly ahead on average
- EEGNet wins 7 of 9 subjects, losing only subjects 2 and 6 - though subject 2 is close enough
  that the result depends on random seed (see "How stable are these results?" below)
- Cross-validating the baseline's own hyperparameters (CSP components, LDA shrinkage) didn't
  improve it - mean accuracy actually dropped slightly to 59.7% (see "Did tuning the baseline
  help?" below)
- Best single result: 79.9% (EEGNet, subject 3)
- 4-class motor imagery, 22 EEG channels, 576 trials/subject total

## Methodology

- **Preprocessing**: raw EEG band-pass filtered to 4-38Hz (covers the mu/beta rhythms relevant
  to motor imagery), epoched to 0-4s relative to cue onset, all 22 channels used.
- **Split**: per subject, session T (288 trials) for training, session E (288 trials) for
  validation - the standard split for this dataset, and harder than a random shuffle since the
  two sessions were recorded on different days.
- **Model**: EEGNet-style CNN (3,444 trainable parameters), PyTorch. Adam optimizer, lr=1e-3,
  batch size 32, cross-entropy loss, up to 100 epochs with early stopping (patience=30) on
  validation loss.
- **Baseline**: CSP (Common Spatial Patterns, 8 components) + LDA - the standard classical
  method for motor imagery EEG, and what the original EEGNet paper itself compares against.
  Same train/val split as the deep learning model. Also tried a cross-validated version that
  tunes `n_components` and LDA shrinkage on the training session only (see below).

**A caveat on what these numbers actually measure**: EEGNet's reported accuracy is the *best*
validation-session score across up to 100 epochs (whichever checkpoint had the highest accuracy
on session E gets saved and reported). The baseline doesn't have an analogous step - it's fit
once on session T and scored once on session E. That's not an apples-to-apples comparison:
picking the best of many looks at the validation session is a real selection advantage that a
single deterministic fit doesn't get. This is also standard practice in most published
EEGNet/CSP+LDA work on this exact dataset (validation-based early stopping, reporting the
selected score), so it's not unusual - but it means these numbers are better read as "best
validation-session performance under each method's own selection procedure" than as an unbiased
estimate of generalization to genuinely unseen data. A stricter protocol would select all
hyperparameters and stopping points using session T alone (e.g. via folds within T), then run a
single frozen evaluation of each method on session E.

## Results

### Accuracy by subject

| Subject | EEGNet | CSP+LDA baseline |
|---|---|---|
| 1 | 74.3% | 69.1% |
| 2 | 37.5% | 48.6% |
| 3 | 79.9% | 73.6% |
| 4 | 57.6% | 54.9% |
| 5 | 41.7% | 35.8% |
| 6 | 36.8% | 46.5% |
| 7 | 66.0% | 62.8% |
| 8 | 76.7% | 74.7% |
| 9 | 77.8% | 76.7% |
| **Mean** | **60.9%** | **60.3%** |

Chance level is 25%.

![Accuracy by subject](results/accuracy_by_subject.png)
![EEGNet vs CSP+LDA baseline](results/comparison_eegnet_vs_baseline.png)

EEGNet slightly outperforms CSP+LDA overall - mean accuracy is 60.9% for EEGNet vs. 60.3% for the
baseline - and wins 7 of the 9 subjects individually. This flipped from an earlier version of
this project where EEGNet lost overall: that version turned out to have a bug (see "Training
curve and an early-stopping mismatch" below) where the early-stopping fix I thought I'd made
wasn't actually being applied when I trained all 9 subjects, so the numbers I was reporting were
generated under the old, buggier setting. Fixing that changed subject 7 from a clear baseline win
to a clear EEGNet win, which was enough to flip the overall average.

My guess for *why* performance is subject-dependent at all: a compact CNN with more free
parameters probably overfits harder than CSP+LDA when the underlying EEG signal is weak, while it
pulls ahead when the signal is stronger. I haven't actually tested that (e.g. varying training set
size, checking train/val gaps per subject), so take it as a hypothesis, not a finding. What the
data does show is that subject 6 is a genuinely hard subject for EEGNet specifically, not just
a fluke of one run - see below.

### How stable are these results across random seeds?

A single accuracy number per subject (one fixed random seed) can make a close result look more
decisive than it is. I reran the 5 subjects with the smallest margins - 2, 5, 6, 7, and 9 - across
5 different seeds to check:

| Subject | Baseline | Reported (seed 42) | Range across 5 seeds | Verdict |
|---|---|---|---|---|
| 2 | 48.6% | 37.5% (loss) | 38.2%-53.5%, median 52.4% | Toss-up - seed 42 happened to be the worst of the 5 seeds. Most seeds actually beat the baseline. |
| 5 | 35.8% | 41.7% (win) | 36.1%-44.1%, median 41.0% | Stable win - every seed beat the baseline. |
| 6 | 46.5% | 36.8% (loss) | 35.8%-49.7%, median 39.9% | Stable loss - only 1 of 5 seeds beat the baseline. |
| 7 | 62.8% | 66.0% (win) | 48.6%-70.1%, median 68.8% | Real win, but noisy - 4 of 5 seeds clustered well above baseline; one seed landed on a genuinely bad local optimum (confirmed by rerunning its full training curve - early stopping worked correctly, it just converged somewhere worse). |
| 9 | 76.7% | 77.8% (win) | 77.8%-81.9%, median 81.6% | Very stable win - every seed beat the baseline comfortably. |

Net effect on the headline: subject 6 is a real, reproducible loss for EEGNet. Subject 2 is
genuinely ambiguous - I'm reporting it as a loss above because that's what seed 42 produced, but
a majority of seeds say otherwise, so I wouldn't read much into that single subject either way.
Subjects 1, 3, 4, and 8 weren't close enough to the baseline to be worth this check.

### Did tuning the baseline help?

The 60.3% baseline above uses fixed, untuned settings (`n_components=8`, no LDA shrinkage), which
isn't obviously fair to compare against a deliberately-designed EEGNet architecture. I added a
cross-validated version (`src/baseline.py --tune`) that picks `n_components` (4-12) and LDA
shrinkage per subject via 5-fold CV on the training session only, never touching the validation
session during selection.

It didn't help - mean accuracy across subjects dropped slightly, from 60.3% to 59.7%, and got
worse specifically for subjects 1, 6, and 7. This isn't a bug in the tuning (the cross-validation
is done correctly, with no leakage from the validation session). My read is that it's a real
property of this dataset: the two sessions were recorded on different days, and EEG signal
statistics are known to drift between recording sessions (a well-documented issue in BCI
research). Hyperparameters chosen by cross-validating within one session don't necessarily
transfer to a different day's recording. So the untuned baseline above isn't lucky - it's roughly
as good as anything a proper search finds, which says something about this dataset more than
about the tuning method.

### Confusion matrices: subject 3 vs subject 6

Comparing EEGNet's best and worst subjects shows the *type* of error is different, not just the
rate.

Subject 6 (36.8%): three of the four classes (`feet`, `left_hand`, `tongue`) sit around
40% recall; `right_hand` is the weak point at 25% recall - barely above chance for that class
specifically. The single biggest source of confusion is `right_hand` trials predicted as
`left_hand` (22 of 72) - right_hand and left_hand get mixed up with each other more than any
other pair. That's the opposite of subject 3's pattern below, where left/right hand were the two
*best*-separated classes. So this isn't uniformly bad classification - it's one class doing much
worse than the rest, and specifically confusable with its mirror-image class, which is a more
interesting (and more accurate) description than "close to random."

![Subject 6 confusion matrix](results/confusion_subject6.png)

Subject 3 (79.9%): the errors have structure. `left_hand` and `right_hand` recall are 91.7% and
94.4% - almost never confused with each other. That's *consistent with* the expected spatial
organization of motor cortex activity (left/right hand imagery activates opposite hemispheres,
which tends to be easier to separate from EEG) - though a single confusion matrix can't prove
that's the mechanism, it's a plausible read given the literature, not a demonstrated explanation.
`feet` is this subject's weakest class at 52.8% recall, confused with all three other classes
about equally - foot motor representation is more medial and less lateralized, which is a harder
case to decode and matches what's reported in the literature. (This is one subject picked to
illustrate the best case in detail, not a claim that it's representative of all nine.)

![Subject 3 confusion matrix](results/confusion_subject3.png)

### Training curve and an early-stopping mismatch

100-epoch run on subject 3, train vs val loss and accuracy:

![Training curve for subject 3](results/training_curve_subject3.png)

Overfitting starts around epoch 30-40: training accuracy keeps climbing toward 90%+ while
validation accuracy flattens around 75-80%.

This also surfaced a mismatch between the early-stopping criterion and the metric I actually
care about: early stopping was tracking validation *loss*, but loss and accuracy don't
necessarily peak at the same epoch. A tighter loss-based stopping threshold cut training off at
epoch 61 and missed a later accuracy peak at epoch 85 (0.757 vs. 0.799 final accuracy). Since
this model trains in under a minute regardless, I made early stopping more lenient
(`patience=30`) rather than tune it to save a few seconds of runtime - the checkpoint logic
already saves whichever epoch had the best validation accuracy, so there's no benefit to cutting
training off aggressively.

Worth being explicit about: I made this change *because* I saw a better peak on session E's
accuracy curve, not as a decision fixed in advance. That's session E informing a training
decision, not just evaluating a frozen one - the same caveat as the checkpoint-selection point
above, just one step earlier in the process. I'm not aware of a way this specific change could
have hurt the baseline's comparison (it only affects how long EEGNet trains), but it's still
worth naming as adaptive reuse of the evaluation session rather than treating patience=30 as if
it were chosen a priori.

That fix only changed `train.py`'s own default, though - the script that trains all 9 subjects
(`run_all_subjects.py`) had `patience=15` hardcoded separately, silently overriding it. So the
first version of the results above was generated without the fix actually being applied. I found
this later while double-checking the comparison against the baseline, fixed the override, and
reran everything - that's the version reported in this README now.

## Why this project

This project explores deep-learning-based decoding of motor-imagery EEG using a compact CNN
architecture designed for EEG signals, benchmarked against a standard classical method. The goal
was hands-on experience with physiological time-series data, a PyTorch implementation of an
EEGNet-style architecture based on the published model, and an honest evaluation of when (and
when not) a deep learning approach actually outperforms a simpler one on a small biomedical
dataset.

## Project structure

```
motor-imagery-eeg-classification/
├── README.md
├── requirements.txt
├── data/                    # MOABB cache (gitignored)
├── checkpoints/             # saved model weights (gitignored)
├── results/                 # plots + csv from the analysis below
├── notebooks/
└── src/
    ├── model.py              # EEGNet
    ├── dataset.py             # loads BCI IV 2a via MOABB
    ├── train.py               # train one subject (EEGNet), CLI entry point
    ├── run_all_subjects.py    # train all 9 subjects, save accuracy comparison
    ├── baseline.py            # CSP+LDA classical baseline, one subject or --all, +optional CV tuning
    ├── compare_baseline.py    # EEGNet vs baseline comparison table + chart
    ├── analyze_subject.py     # confusion matrix for a saved checkpoint
    ├── seed_stability.py      # re-runs a subject across multiple seeds to check result stability
    └── utils.py               # seeding, device selection, early stopping
```

## Setup

```bash
cd ~/Documents/GitHub/motor-imagery-eeg-classification
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Apple Silicon, `train.py` uses the `mps` backend automatically. Dataset downloads
automatically on first run (via MOABB, cached after that).

## Reproducing the results

```bash
python -m src.train --subject 3 --epochs 100 --plot
python -m src.run_all_subjects
python -m src.baseline --all
python -m src.baseline --all --tune
python -m src.compare_baseline
python -m src.analyze_subject --subject 6
python -m src.seed_stability --subjects 2,5,6,7,9 --seeds 0,1,2,3,4
```

## Possible extensions

- A stricter evaluation protocol: select hyperparameters and stopping points using only session
  T (e.g. folds within it), then run one frozen evaluation of each method on session E, instead
  of the current approach where E influences checkpoint selection (and, once, the patience
  value itself). See the caveat in Methodology.
- Leave-one-subject-out generalization: train on 8 subjects, test on the 9th, to see whether a
  model can generalize across people instead of being trained per-subject.
- Test the overfitting hypothesis directly (e.g. vary training set size, compare train/val gap
  per subject) instead of leaving it as a guess.
- Extend the seed-stability check to the remaining subjects (1, 3, 4, 8) for completeness.
- Dig into *why* baseline tuning didn't transfer across sessions - e.g. whether per-session
  recalibration (a common approach in real BCI systems) closes the gap.
- Different frequency band (`fmin`/`fmax` in `dataset.py`) or epoch window (`tmin`/`tmax`).
- EEGNet hyperparameters (`F1`/`D`/`F2`/dropout in `model.py`).
- Early stopping on val accuracy instead of val loss.

## Troubleshooting

- Install issues: make sure the venv is activated (`which python` -> `.venv/bin/python`) before
  `pip install`.
- `ModuleNotFoundError: No module named 'src'`: run `python -m src.train ...` from the project
  root, not `python src/train.py`.
