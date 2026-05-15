"""
MFPT Bearing Dataset Loader
Mechanical Failure Prevention Technology Society
"""
import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.signal_processing import segment_signal, normalize_signal, generate_spectrogram


def load_mfpt_data(data_root, signal_length=8192, overlap=0.5):
    """Load all MFPT bearing data.

    Args:
        data_root: Path to MFPT data directory
        signal_length: Length of each signal segment
        overlap: Overlap ratio for segmentation

    Returns:
        signals: List of 1D numpy arrays
        labels: List of integer labels
        class_names: List of class name strings
    """
    from src.config import MFPT_CLASSES, MFPT_CLASS_MAP

    signals = []
    labels = []

    # Map directory names to classes
    dir_class_map = {
        "1 - Three Baseline Conditions": "Baseline",
        "2 - Three Outer Race Fault Conditions": "OuterRace",
        "3 - Seven More Outer Race Fault Conditions": "OuterRace",
        "4 - Seven Inner Race Fault Conditions": "InnerRace",
    }

    for dir_name, class_name in dir_class_map.items():
        dir_path = os.path.join(data_root, dir_name)
        if not os.path.exists(dir_path):
            continue

        for f in sorted(os.listdir(dir_path)):
            if not f.endswith('.mat'):
                continue
            fpath = os.path.join(dir_path, f)
            try:
                mat = sio.loadmat(fpath)
                if 'bearing' in mat:
                    bearing = mat['bearing'][0, 0]
                    # Extract vibration signal from 'gs' field
                    if 'gs' in bearing.dtype.names:
                        sig = bearing['gs'].flatten()
                        # Get sampling rate
                        sr = 97656  # Default MFPT sampling rate
                        if 'sr' in bearing.dtype.names:
                            sr = int(bearing['sr'].flatten()[0])

                        segments = segment_signal(sig, signal_length, overlap)
                        signals.extend(segments)
                        labels.extend([MFPT_CLASS_MAP[class_name]] * len(segments))
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

    print(f"MFPT: Loaded {len(signals)} segments, {len(MFPT_CLASSES)} classes")
    for cls_name, cls_idx in MFPT_CLASS_MAP.items():
        count = sum(1 for l in labels if l == cls_idx)
        print(f"  {cls_name}: {count} segments")

    return signals, labels, MFPT_CLASSES


class MFPTDataset(Dataset):
    """PyTorch Dataset for MFPT bearing data."""

    def __init__(self, signals, labels, fs=97656, spec_size=(224, 224),
                 transform=None):
        self.signals = signals
        self.labels = labels
        self.fs = fs
        self.spec_size = spec_size
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx]
        label = self.labels[idx]

        sig = normalize_signal(sig)
        spec = generate_spectrogram(sig, fs=self.fs, target_size=self.spec_size)
        spec_rgb = np.stack([spec, spec, spec], axis=0)

        if self.transform:
            spec_rgb = self.transform(spec_rgb)

        import torch
        return torch.FloatTensor(spec_rgb), torch.LongTensor([label])[0]
