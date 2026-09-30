# Motor Imagery EEG Classification

Classifying imagined left-hand, right-hand, feet, and tongue movements from EEG recordings using an EEGNet-style CNN and a CSP+LDA baseline.

## Overview

I built this project to explore EEG classification and compare a compact neural network with a classical machine learning approach. I was interested in how the models performed across participants and how much the results changed between training runs.

## Data and methods

I used BCI Competition IV Dataset 2a through MOABB. The dataset contains nine participants, 22 EEG channels, and two recording sessions per participant.

- **Preprocessing:** 4–38 Hz band-pass filtering and four-second epochs.
- **Split:** session T for training and session E for validation, with a separate model for each participant.
- **EEGNet-style CNN:** implemented in PyTorch, based on the published architecture, with 3,444 trainable parameters.
- **CSP+LDA:** eight Common Spatial Patterns components followed by Linear Discriminant Analysis.

The CNN was trained with Adam, a learning rate of 0.001, batch size 32, and a maximum of 100 epochs. Early stopping monitored validation loss, while checkpoint selection used validation accuracy.

## Results

| Method | Mean validation accuracy |
|---|---:|
| EEGNet, seed 42 | 64.2% |
| EEGNet, averaged across seeds 0–4 | 62.0% |
| CSP+LDA, fixed settings | 60.3% |
| CSP+LDA, settings selected through training-session cross-validation | 59.7% |

Each mean includes all nine participants. The five-seed EEGNet result includes 45 runs.

![Validation accuracy by participant, EEGNet seed 42 versus CSP+LDA](results/comparison_eegnet_vs_baseline.png)

EEGNet had a higher average validation accuracy, but the difference depended on the participant and random seed. Subject 2, for example, scored 55.2% with seed 42, while the five additional runs ranged from 35.1% to 56.2%.

Tuning the baseline’s component count and LDA shrinkage within the training session did not improve its average validation score in this experiment.

## Limitations

Session E was used to select EEGNet’s best checkpoint as well as report its accuracy. The fixed baseline did not have the same selection step, so these results do not establish that EEGNet generalizes better.

A next step is to select training settings using session T alone before evaluating on E, while acknowledging that E has already informed development. The current models are also trained separately for each participant; they have not been tested on unseen participants.

## Implementation notes

During development, I corrected an early-stopping setting that differed between scripts and changed the model's dummy shape-inference pass to evaluation mode. Checkpoints and confusion-matrix files now include the seed to keep runs separate.

I also added the five-seed comparison after seeing how much individual results could vary. More complete experiment records, including environment versions and saved predictions, would make future comparisons easier to track.

## Running the project

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train one participant:

```bash
python -m src.train --subject 3 --epochs 100 --plot
```

Run both methods across all participants:

```bash
python -m src.run_all_subjects
python -m src.baseline --all
python -m src.compare_baseline
```

Run baseline tuning and the seed comparison:

```bash
python -m src.baseline --all --tune
python -m src.seed_stability --subjects 1,2,3,4,5,6,7,8,9 --seeds 0,1,2,3,4
```

Generate a confusion matrix from a saved checkpoint:

```bash
python -m src.analyze_subject --subject 3 --seed 42
```

MOABB downloads the dataset on first use. Results may vary across hardware and package versions.

## References

- [BCI Competition IV](https://www.bbci.de/competition/iv/)
- [MOABB](https://moabb.neurotechx.com/)
- [EEGNet paper](https://arxiv.org/abs/1611.08024)
- [EEGNet reference implementation](https://github.com/vlawhern/arl-eegmodels)
