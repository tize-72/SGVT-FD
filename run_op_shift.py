"""
Operating Condition Shift Experiment.
Train on CWRU 0HP+1HP, test on 2HP+3HP (and vice versa).
"""
import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import *
from src.utils.signal_processing import segment_signal, normalize_signal, generate_spectrogram
from src.models.sgvt import SGVTModel
from src.models.baselines import CVTBaseline, ViTBaseline
from src.models.sgvt import NoGroupingBaseline
from src.utils.metrics import compute_metrics

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "supplementary", "op_shift")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_crwu_by_load(data_root, signal_length=8192, overlap=0.5):
    """Load CWRU data with load condition tracking.
    Returns dict: load -> (signals, labels)
    """
    from src.config import CRWU_CLASS_MAP

    load_data = {0: [], 1: [], 2: [], 3: []}
    de_key = "_DE_time"

    # Normal baseline
    normal_dir = os.path.join(data_root, "Normal Baseline")
    if os.path.exists(normal_dir):
        for f in sorted(os.listdir(normal_dir)):
            if f.endswith('.mat'):
                # Extract load from filename suffix before .mat
                load = int(f.replace('.mat', '').split('_')[-1])
                mat = loadmat(os.path.join(normal_dir, f))
                for key in mat:
                    if key.endswith(de_key):
                        sig = mat[key].flatten()
                        segments = segment_signal(sig, signal_length, overlap)
                        for seg in segments:
                            load_data[load].append((seg, CRWU_CLASS_MAP["Normal"]))

    # Fault data
    fault_base = os.path.join(data_root, "12k Drive End Bearing Fault Data")
    if not os.path.exists(fault_base):
        fault_base = os.path.join(data_root, "12k Fan End Bearing Fault Data")

    fault_dirs = {
        "Ball": "Ball", "Inner Race": "InnerRace",
        "Outer Race": "OuterRace", "InnerRace": "InnerRace", "OuterRace": "OuterRace"
    }

    for fault_dir_name, fault_class in fault_dirs.items():
        fault_dir = os.path.join(fault_base, fault_dir_name)
        if not os.path.exists(fault_dir):
            continue

        # Process subdirectories (size dirs like 0007, 0014, ...)
        for item in sorted(os.listdir(fault_dir)):
            item_path = os.path.join(fault_dir, item)
            if os.path.isdir(item_path):
                for f in sorted(os.listdir(item_path)):
                    if f.endswith('.mat'):
                        # Extract load: filename looks like IR007_0.mat, B007_1.mat, etc.
                        load = int(f.replace('.mat', '').split('_')[-1])
                        mat = loadmat(os.path.join(item_path, f))
                        for key in mat:
                            if key.endswith(de_key):
                                sig = mat[key].flatten()
                                segments = segment_signal(sig, signal_length, overlap)
                                for seg in segments:
                                    load_data[load].append((seg, CRWU_CLASS_MAP[fault_class]))
            else:
                # Direct files in fault dir (Outer Race sometimes has files directly)
                if item.endswith('.mat'):
                    load = int(item.replace('.mat', '').split('_')[-1])
                    try:
                        mat = loadmat(item_path)
                        for key in mat:
                            if key.endswith(de_key):
                                sig = mat[key].flatten()
                                segments = segment_signal(sig, signal_length, overlap)
                                for seg in segments:
                                    load_data[load].append((seg, CRWU_CLASS_MAP[fault_class]))
                    except:
                        pass

        # Subdirs: Centered, Opposite, Orthogonal (for Outer Race)
        for subdir in ["Centered", "Opposite", "Orthogonal"]:
            subdir_path = os.path.join(fault_dir, subdir)
            if not os.path.exists(subdir_path):
                continue
            for size_dir in sorted(os.listdir(subdir_path)):
                size_path = os.path.join(subdir_path, size_dir)
                if not os.path.isdir(size_path):
                    continue
                for f in sorted(os.listdir(size_path)):
                    if f.endswith('.mat'):
                        load = int(f.replace('.mat', '').split('_')[-1])
                        try:
                            mat = loadmat(os.path.join(size_path, f))
                            for key in mat:
                                if key.endswith(de_key):
                                    sig = mat[key].flatten()
                                    segments = segment_signal(sig, signal_length, overlap)
                                    for seg in segments:
                                        load_data[load].append((seg, CRWU_CLASS_MAP[fault_class]))
                        except:
                            pass

    # Print stats
    total = 0
    for load, items in load_data.items():
        print(f"  Load {load}HP: {len(items)} segments")
        total += len(items)
    print(f"  Total: {total} segments")

    return load_data


class CWRULoadDataset(Dataset):
    """Dataset for CWRU data with spectrogram conversion."""
    def __init__(self, items, fs=12000, spec_size=(224, 224)):
        self.items = items  # list of (signal, label)
        self.fs = fs
        self.spec_size = spec_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sig, label = self.items[idx]
        sig = normalize_signal(sig)
        spec = generate_spectrogram(sig, fs=self.fs, target_size=self.spec_size)
        spec_rgb = np.stack([spec, spec, spec], axis=0)
        return torch.FloatTensor(spec_rgb), torch.LongTensor([label])[0]


def train_model(model, train_loader, val_loader, test_loader, n_epochs=50, model_name="model"):
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=n_epochs - 5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    best_val_acc = 0
    out_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, loss, _ = model(images, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _, _ = model(images, labels)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_metrics = compute_metrics(val_labels, val_preds)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
        scheduler.step()

    # Load best and evaluate on test
    if os.path.exists(os.path.join(out_dir, "best_model.pt")):
        model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), weights_only=True))

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits, _, _ = model(images, labels)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_metrics = compute_metrics(test_labels, test_preds)
    params = sum(p.numel() for p in model.parameters())
    return test_metrics, params


def run_op_shift():
    """Run operating condition shift experiments."""
    print("=" * 70)
    print("Operating Condition Shift Experiment")
    print("=" * 70)

    print("\nLoading CWRU data with load condition labels...")
    load_data = load_crwu_by_load(CRWU_ROOT)

    seeds = [42, 123, 456]
    experiments = {
        'low2high': {
            'train_loads': [0, 1],
            'test_loads': [2, 3],
            'desc': 'Train on 0HP+1HP → Test on 2HP+3HP'
        },
        'high2low': {
            'train_loads': [2, 3],
            'test_loads': [0, 1],
            'desc': 'Train on 2HP+3HP → Test on 0HP+1HP'
        },
    }

    all_results = {}

    for exp_name, exp_cfg in experiments.items():
        print(f"\n--- {exp_cfg['desc']} ---")

        # Build datasets
        train_items = []
        for l in exp_cfg['train_loads']:
            train_items.extend(load_data[l])
        test_items = []
        for l in exp_cfg['test_loads']:
            test_items.extend(load_data[l])

        print(f"  Train: {len(train_items)} segments ({exp_cfg['train_loads']}HP)")
        print(f"  Test: {len(test_items)} segments ({exp_cfg['test_loads']}HP)")

        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)

            random.seed(seed)
            random.shuffle(train_items)
            val_size = int(0.15 * len(train_items))
            train_data = train_items[:-val_size]
            val_data = train_items[-val_size:]

            train_ds = CWRULoadDataset(train_data)
            val_ds = CWRULoadDataset(val_data)
            test_ds = CWRULoadDataset(test_items)

            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
            test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

            # SGVT-FD
            print(f"    SGVT-FD (seed={seed})...")
            sgvt = SGVTModel(num_classes=4, num_groups=32, use_domain_prior=False,
                              use_mi_loss=True, device=DEVICE).to(DEVICE)
            metrics, params = train_model(sgvt, train_loader, val_loader, test_loader,
                                          model_name=f"{exp_name}_sgvt_seed{seed}")
            all_results[f"{exp_name}_sgvt_seed{seed}"] = {
                'accuracy': metrics['accuracy'], 'f1': metrics['f1'], 'params': params
            }
            print(f"      Acc: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1']*100:.2f}%")

            # CVT
            print(f"    CVT (seed={seed})...")
            cvt = CVTBaseline(num_classes=4, num_groups=32, device=DEVICE).to(DEVICE)
            metrics, params = train_model(cvt, train_loader, val_loader, test_loader,
                                          model_name=f"{exp_name}_cvt_seed{seed}")
            all_results[f"{exp_name}_cvt_seed{seed}"] = {
                'accuracy': metrics['accuracy'], 'f1': metrics['f1'], 'params': params
            }
            print(f"      Acc: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1']*100:.2f}%")

            # No Grouping
            print(f"    No Grouping (seed={seed})...")
            ng = NoGroupingBaseline(num_classes=4, device=DEVICE).to(DEVICE)
            metrics, params = train_model(ng, train_loader, val_loader, test_loader,
                                          model_name=f"{exp_name}_nogroup_seed{seed}")
            all_results[f"{exp_name}_nogroup_seed{seed}"] = {
                'accuracy': metrics['accuracy'], 'f1': metrics['f1'], 'params': params
            }
            print(f"      Acc: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1']*100:.2f}%")

    # Save results
    with open(os.path.join(OUTPUT_DIR, "op_shift_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("-" * 70)
    for exp_name in experiments:
        for method in ['sgvt', 'cvt', 'nogroup']:
            accs = []
            for seed in seeds:
                key = f"{exp_name}_{method}_seed{seed}"
                if key in all_results:
                    accs.append(all_results[key]['accuracy'] * 100)
            if accs:
                print(f"  {exp_name}_{method}: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")

    print(f"\nResults saved to {OUTPUT_DIR}/op_shift_results.json")
    return all_results


if __name__ == "__main__":
    run_op_shift()
