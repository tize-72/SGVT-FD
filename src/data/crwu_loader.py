"""
CRWU Bearing Dataset Loader
Case Western Reserve University Bearing Data Center
"""
import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.signal_processing import segment_signal, normalize_signal, generate_spectrogram


def load_crwu_data(data_root, signal_length=8192, overlap=0.5, use_de=True):
    """Load all CRWU bearing data.

    Args:
        data_root: Path to CRWU data directory
        signal_length: Length of each signal segment
        overlap: Overlap ratio for segmentation
        use_de=True: Use Drive End (DE) accelerometer data

    Returns:
        signals: List of 1D numpy arrays
        labels: List of integer labels
        class_names: List of class name strings
    """
    from src.config import CRWU_CLASSES, CRWU_CLASS_MAP, CRWU_FAULT_SIZES

    signals = []
    labels = []

    # Determine which key to use
    de_key_suffix = "_DE_time" if use_de else "_FE_time"

    # 1. Normal baseline
    normal_dir = os.path.join(data_root, "Normal Baseline")
    if os.path.exists(normal_dir):
        for f in sorted(os.listdir(normal_dir)):
            if f.endswith('.mat'):
                mat = sio.loadmat(os.path.join(normal_dir, f))
                for key in mat:
                    if key.endswith(de_key_suffix):
                        sig = mat[key].flatten()
                        segments = segment_signal(sig, signal_length, overlap)
                        signals.extend(segments)
                        labels.extend([CRWU_CLASS_MAP["Normal"]] * len(segments))

    # 2. Fault data - 12k Drive End
    fault_base = os.path.join(data_root, "12k Drive End Bearing Fault Data")
    if not os.path.exists(fault_base):
        fault_base = os.path.join(data_root, "12k Fan End Bearing Fault Data")

    fault_types = {
        "Ball": "Ball",
        "Inner Race": "InnerRace",
        "InnerRace": "InnerRace",
        "Outer Race": "OuterRace",
        "OuterRace": "OuterRace",
    }

    for fault_dir_name, fault_class in fault_types.items():
        fault_dir = os.path.join(fault_base, fault_dir_name)
        if not os.path.exists(fault_dir):
            continue

        for size_dir in CRWU_FAULT_SIZES:
            size_path = os.path.join(fault_dir, size_dir)
            if not os.path.exists(size_path):
                continue

            for f in sorted(os.listdir(size_path)):
                if f.endswith('.mat'):
                    fpath = os.path.join(size_path, f)
                    try:
                        mat = sio.loadmat(fpath)
                        for key in mat:
                            if key.endswith(de_key_suffix):
                                sig = mat[key].flatten()
                                segments = segment_signal(sig, signal_length, overlap)
                                signals.extend(segments)
                                labels.extend([CRWU_CLASS_MAP[fault_class]] * len(segments))
                    except Exception as e:
                        print(f"Error loading {fpath}: {e}")

        # Check for subdirectories (Outer Race has Centered/Opposite/Orthogonal)
        for subdir in ["Centered", "Opposite", "Orthogonal"]:
            subdir_path = os.path.join(fault_dir, subdir)
            if not os.path.exists(subdir_path):
                continue
            for size_dir in CRWU_FAULT_SIZES:
                size_path = os.path.join(subdir_path, size_dir)
                if not os.path.exists(size_path):
                    continue
                for f in sorted(os.listdir(size_path)):
                    if f.endswith('.mat'):
                        fpath = os.path.join(size_path, f)
                        try:
                            mat = sio.loadmat(fpath)
                            for key in mat:
                                if key.endswith(de_key_suffix):
                                    sig = mat[key].flatten()
                                    segments = segment_signal(sig, signal_length, overlap)
                                    signals.extend(segments)
                                    labels.extend([CRWU_CLASS_MAP[fault_class]] * len(segments))
                        except Exception as e:
                            print(f"Error loading {fpath}: {e}")

    print(f"CRWU: Loaded {len(signals)} segments, {len(CRWU_CLASSES)} classes")
    for cls_name, cls_idx in CRWU_CLASS_MAP.items():
        count = sum(1 for l in labels if l == cls_idx)
        print(f"  {cls_name}: {count} segments")

    return signals, labels, CRWU_CLASSES


class CRWUDataset(Dataset):
    """PyTorch Dataset for CRWU bearing data.

    Returns spectrogram images and labels.
    """

    def __init__(self, signals, labels, fs=12000, spec_size=(224, 224),
                 transform=None, augment=False):
        self.signals = signals
        self.labels = labels
        self.fs = fs
        self.spec_size = spec_size
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx]
        label = self.labels[idx]

        # Normalize signal
        sig = normalize_signal(sig)

        # Generate spectrogram
        spec = generate_spectrogram(sig, fs=self.fs, target_size=self.spec_size)

        # Convert to 3-channel (RGB) for VLM input
        spec_rgb = np.stack([spec, spec, spec], axis=0)  # (3, H, W)

        if self.transform:
            spec_rgb = self.transform(spec_rgb)

        import torch
        return torch.FloatTensor(spec_rgb), torch.LongTensor([label])[0]
