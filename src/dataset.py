"""
Loads the BCI Competition IV Dataset 2a via MOABB (MOABB calls it BNCI2014_001 -
it's the same dataset: 9 subjects, 22 EEG channels, 4-class motor imagery:
left hand, right hand, feet, tongue).

MOABB downloads the raw data to ~/mne_data/ on first use and caches it after that.
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


@dataclass
class EEGData:
    X: np.ndarray          # (n_trials, n_channels, n_samples)
    y: np.ndarray          # (n_trials,) integer labels
    label_names: list      # index -> class name
    sessions: np.ndarray   # (n_trials,) session id string, e.g. "0train" / "1test"


def load_subject_data(
    subject_id: int,
    fmin: float = 4.0,
    fmax: float = 38.0,
    tmin: float = 0.0,
    tmax: float = 4.0,
) -> EEGData:
    """
    Load and band-pass filter one subject's motor imagery trials.

    fmin/fmax: frequency band in Hz. 4-38 Hz covers the mu (8-13Hz) and beta
        (13-30Hz) rhythms most relevant to motor imagery.
    tmin/tmax: seconds relative to cue onset to epoch (0-4s is the standard
        window for this dataset's cue-based trials).
    """
    # imported lazily so `python -m src.model` etc. don't require moabb installed
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return EEGData(
        X=X.astype(np.float32),
        y=y_encoded.astype(np.int64),
        label_names=list(encoder.classes_),
        sessions=metadata["session"].to_numpy(),
    )


class EEGDataset(Dataset):
    """Thin torch Dataset wrapper around an EEGData's arrays (or a subset of them)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def session_split(data: EEGData):
    """
    Standard BCI IV 2a evaluation: train on session 'T' (0train), test on
    session 'E' (1test) - a harder, more realistic split than a random
    shuffle, since it tests generalization to a different day's recording.
    Falls back to a random 80/20 split if session labels look different.
    """
    train_mask = np.array(["train" in s.lower() or s.endswith("T") for s in data.sessions])
    if train_mask.sum() == 0 or train_mask.sum() == len(train_mask):
        # fallback: random split
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(data.y))
        split = int(0.8 * len(idx))
        train_idx, test_idx = idx[:split], idx[split:]
    else:
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(~train_mask)[0]

    train_ds = EEGDataset(data.X[train_idx], data.y[train_idx])
    test_ds = EEGDataset(data.X[test_idx], data.y[test_idx])
    return train_ds, test_ds
