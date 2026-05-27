"""Public minimal dataset and window-construction utilities."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_arrays(data_dir):
    data_dir = Path(data_dir)
    required = ["X.npy", "y.npy", "groups.npy", "sessions.npy"]
    for fname in required:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(f"{data_dir / fname} not found")

    x = np.load(data_dir / "X.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy")
    groups = np.load(data_dir / "groups.npy", allow_pickle=True)
    sessions = np.load(data_dir / "sessions.npy", allow_pickle=True)
    epoch_ids = np.load(data_dir / "epoch_ids.npy") if (data_dir / "epoch_ids.npy").exists() else None
    return {"X": x, "y": y, "groups": groups, "sessions": sessions, "epoch_ids": epoch_ids}


def build_windows(data, indices, seq_len=21):
    indices = np.sort(indices)
    sessions = data["sessions"]
    epoch_ids = data.get("epoch_ids", None)
    windows = []

    for sess in np.unique(sessions[indices]):
        ids = np.sort(indices[sessions[indices] == sess])
        if len(ids) < seq_len:
            continue
        if epoch_ids is not None:
            epoch_ids_sub = epoch_ids[ids]
            for i in range(len(ids) - seq_len + 1):
                if np.all(np.diff(epoch_ids_sub[i:i + seq_len]) == 1):
                    windows.append(ids[i:i + seq_len])
        else:
            for i in range(len(ids) - seq_len + 1):
                windows.append(ids[i:i + seq_len])

    return np.array(windows, dtype=np.int64)


class SequenceDataset(Dataset):
    def __init__(self, x, y, windows):
        self.x = x
        self.y = y
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        win = self.windows[idx]
        sample = self.x[win].copy().astype(np.float32)
        labels = self.y[win].astype(np.int64)
        mean = sample.mean(axis=-1, keepdims=True)
        std = sample.std(axis=-1, keepdims=True) + 1e-8
        sample = (sample - mean) / std
        return torch.from_numpy(sample), torch.from_numpy(labels)
