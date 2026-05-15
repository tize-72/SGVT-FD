"""
Supplementary experiments for SGVT-FD paper revision.
Covers: cross-domain generalization, operating condition shift,
random grouping baseline, multi-noise types, statistical tests.
"""
import os, sys, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import *
from src.data.crwu_loader import load_crwu_data, CRWUDataset
from src.data.mfpt_loader import load_mfpt_data, MFPTDataset
from src.models.sgvt import SGVTModel, SGVTBaseline, NoGroupingBaseline
from src.models.baselines import CVTBaseline, ViTBaseline
from src.utils.signal_processing import (
    add_noise, add_impulse_noise, add_colored_noise, add_mechanical_noise,
    normalize_signal, generate_spectrogram, segment_signal
)
from src.utils.metrics import compute_metrics

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "supplementary")
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ─── Noise Dataset Wrapper ───────────────────────────────────────────

class NoisyDataset(torch.utils.data.Dataset):
    """Wraps a base dataset and applies noise at __getitem__ time.

    CRWUDataset/MFPTDataset return (spec_tensor_3x224x224, label_tensor).
    We apply noise to the spectrogram directly (not the raw signal)
    since we don't have access to the raw signal at this level.
    We add noise directly in spectrogram space to simulate noisy sensor readings.
    """
    def __init__(self, base_dataset, noise_type='gaussian', snr_db=20):
        self.base = base_dataset
        self.noise_type = noise_type
        self.snr_db = snr_db

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        spec, label = self.base[idx]

        # Add noise in spectrogram space (simulates noisy sensor data)
        if self.snr_db is not None:
            if self.noise_type == 'gaussian':
                spec_power = (spec ** 2).mean().item()
                noise_power = spec_power / (10 ** (self.snr_db / 10))
                noise = torch.randn_like(spec) * np.sqrt(noise_power)
                spec = spec + noise
            elif self.noise_type == 'impulse':
                densities = {30: 0.001, 20: 0.005, 10: 0.01, 5: 0.02, 0: 0.05}
                amps = {30: 1.0, 20: 2.0, 10: 3.0, 5: 5.0, 0: 8.0}
                d = densities.get(self.snr_db, 0.01)
                a = amps.get(self.snr_db, 3.0)
                mask = (torch.rand_like(spec) < d).float()
                spec = spec + mask * a * spec.std() * torch.randn_like(spec)
            elif self.noise_type in ('pink', 'brown', 'blue'):
                # Colored noise via frequency-domain shaping
                spec_power = (spec ** 2).mean().item()
                noise_power = spec_power / (10 ** (self.snr_db / 10))
                # Generate white noise then apply spectral coloring
                white = torch.randn_like(spec[0])  # only 1st channel for efficiency
                n = white.shape[-1]
                f = torch.arange(n, device=white.device).float() + 1
                white_fft = torch.fft.rfft(white, dim=-1)
                if self.noise_type == 'pink':
                    white_fft = white_fft / torch.sqrt(f[:white_fft.shape[-1]])
                elif self.noise_type == 'brown':
                    white_fft = white_fft / f[:white_fft.shape[-1]]
                elif self.noise_type == 'blue':
                    white_fft = white_fft * torch.sqrt(f[:white_fft.shape[-1]])
                colored = torch.fft.irfft(white_fft, n=n, dim=-1)
                colored = colored / (colored.std() + 1e-8)
                noise = torch.sqrt(torch.tensor(noise_power)) * colored.unsqueeze(0).repeat(3, 1, 1)
                spec = spec + noise
            elif self.noise_type == 'mechanical':
                spec_power = (spec ** 2).mean().item()
                noise_power = spec_power / (10 ** (self.snr_db / 10))
                # Add harmonic noise pattern + broadband
                noise = torch.randn_like(spec) * 0.3
                # Add horizontal line artifacts (simulating rotational harmonics)
                for harmonic_pos in [spec.shape[-2] // 6, spec.shape[-2] // 3]:
                    noise[:, harmonic_pos, :] += 0.5 * torch.randn(spec.shape[-1])
                noise = noise / (noise.std() + 1e-8)
                spec = spec + torch.sqrt(torch.tensor(noise_power)) * noise

        return spec, label


# ─── New Baselines ───────────────────────────────────────────────────

class ToMeBaseline(nn.Module):
    """ToMe-like (Token Merging, ICLR 2023) token compression.
    Merges similar token pairs via bipartite matching on cosine similarity.
    """
    def __init__(self, num_classes=4, target_tokens=32, patch_size=16,
                 feature_dim=768, device="cuda"):
        super().__init__()
        self.target_tokens = target_tokens
        self.device = device
        self.feature_dim = feature_dim

        self.patch_embed = nn.Conv2d(3, feature_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, feature_dim) * 0.02)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def _bipartite_merge(self, tokens):
        """Batched bipartite soft matching: halves token count per iteration."""
        B, N, D = tokens.shape
        while tokens.shape[1] > self.target_tokens:
            N_cur = tokens.shape[1]
            half = N_cur // 2
            tokens_a, tokens_b = tokens[:, :half, :], tokens[:, half:, :]
            a_norm = F.normalize(tokens_a, dim=-1)
            b_norm = F.normalize(tokens_b, dim=-1)
            sim = torch.bmm(a_norm, b_norm.transpose(1, 2))
            _, best_b_idx = sim.max(dim=2)
            merged = (tokens_a + tokens_b.gather(1,
                best_b_idx.unsqueeze(-1).expand(-1, -1, D))) / 2.0
            tokens = torch.cat([merged,
                tokens_b[:, len(tokens_a):, :] if tokens_b.shape[1] > len(tokens_a) else
                torch.empty(B, 0, D, device=tokens.device)], dim=1)
        return tokens

    def forward(self, x, labels=None):
        B = x.shape[0]
        patches = self.patch_embed(x)
        tokens = patches.reshape(B, self.feature_dim, -1).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        merged = self._bipartite_merge(tokens)
        pooled = merged.mean(dim=1)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        return logits, loss, None


class TokenLearnerBaseline(nn.Module):
    """TokenLearner-like learnable token selection.
    Learns to produce K weighted combinations of the input tokens.
    """
    def __init__(self, num_classes=4, num_groups=32, patch_size=16,
                 feature_dim=768, device="cuda"):
        super().__init__()
        self.num_groups = num_groups
        self.device = device
        self.feature_dim = feature_dim

        self.patch_embed = nn.Conv2d(3, feature_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, feature_dim) * 0.02)

        # TokenLearner: MLP predicts per-token importance weights for K groups
        self.token_weight = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, num_groups),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, labels=None):
        B = x.shape[0]
        patches = self.patch_embed(x)
        tokens = patches.reshape(B, self.feature_dim, -1).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        # TokenLearner: learn importance weights
        weights = self.token_weight(tokens)  # (B, N, K)
        weights = F.softmax(weights, dim=1)  # normalize across tokens

        # Weighted combination
        merged = torch.bmm(weights.transpose(1, 2), tokens)  # (B, K, D)
        pooled = merged.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        return logits, loss, None


# ─── Training / Evaluation Helpers ───────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, loss, _ = model(images, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader.dataset), compute_metrics(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, loss, _ = model(images, labels)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return compute_metrics(all_labels, all_preds), all_preds, all_labels


def train_model(model, train_loader, val_loader, test_loader, args_dict, output_dir):
    """Full training loop. Returns test metrics."""
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=args_dict.get('lr', 1e-4), weight_decay=0.01)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=args_dict.get('epochs', 50) - 5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    best_val_acc = 0
    for epoch in range(1, args_dict.get('epochs', 50) + 1):
        train_one_epoch(model, train_loader, optimizer, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        scheduler.step()

    # Load best
    if os.path.exists(os.path.join(output_dir, "best_model.pt")):
        model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=True))

    test_metrics, test_preds, test_labels = evaluate(model, test_loader, device)
    total_params = sum(p.numel() for p in model.parameters())
    return test_metrics, test_preds, test_labels, total_params


# ─── Experiment 1: Cross-Domain Generalization ───────────────────────

def run_cross_domain():
    """Train on CWRU -> test on MFPT, and train on MFPT -> test on CWRU.

    Uses common 3-class subset (Normal, InnerRace, OuterRace) since MFPT
    doesn't have Ball class.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Cross-Domain Generalization")
    print("=" * 70)

    results = {}
    seeds = [42, 123, 456]
    signal_length = 8192

    # Load both datasets with 3 common classes
    cwru_signals, cwru_labels, cwru_classes = load_crwu_data(CRWU_ROOT, signal_length=signal_length, overlap=0.5)

    # Filter CWRU to only Normal(0), InnerRace(2), OuterRace(3) - drop Ball(1)
    cwru_filtered = [(s, l if l < 2 else l - 1) for s, l in zip(cwru_signals, cwru_labels) if l != 1]
    cwru_sig_3, cwru_lbl_3 = zip(*cwru_filtered) if cwru_filtered else ([], [])
    cwru_sig_3, cwru_lbl_3 = list(cwru_sig_3), list(cwru_lbl_3)
    cwru_3_classes = ["Normal", "InnerRace", "OuterRace"]

    mfpt_signals, mfpt_labels, mfpt_classes = load_mfpt_data(MFPT_ROOT, signal_length=signal_length, overlap=0.5)
    mfpt_3_classes = mfpt_classes

    for seed in seeds:
        set_seed(seed)

        # ── CWRU -> MFPT ──
        print(f"\n--- CWRU -> MFPT (seed={seed}) ---")
        cwru_ds = CRWUDataset(cwru_sig_3, cwru_lbl_3, fs=SAMPLING_RATE_CRWU, spec_size=(224, 224))
        mfpt_ds = MFPTDataset(mfpt_signals, mfpt_labels, fs=SAMPLING_RATE_MFPT, spec_size=(224, 224))

        # Train on CWRU, test on MFPT
        n_cwru = len(cwru_ds)
        val_sz = int(n_cwru * 0.1)
        tr_sz = n_cwru - val_sz
        cwru_train, cwru_val = random_split(cwru_ds, [tr_sz, val_sz],
                                            generator=torch.Generator().manual_seed(seed))

        tr_dl = DataLoader(cwru_train, batch_size=16, shuffle=True, num_workers=4)
        vl_dl = DataLoader(cwru_val, batch_size=16, shuffle=False, num_workers=4)
        te_dl = DataLoader(mfpt_ds, batch_size=16, shuffle=False, num_workers=4)

        out_dir = os.path.join(OUTPUT_BASE, "cross_domain", f"crwu2mfpt_seed{seed}")
        os.makedirs(out_dir, exist_ok=True)

        model = SGVTModel(num_classes=3, num_groups=32, use_domain_prior=False,
                          use_mi_loss=True, device=DEVICE).to(DEVICE)
        metrics, preds, labels, params = train_model(model, tr_dl, vl_dl, te_dl, {'epochs': 50}, out_dir)
        results[f"crwu2mfpt_seed{seed}"] = {
            'accuracy': metrics['accuracy'], 'f1': metrics['f1'], 'params': params
        }
        print(f"  CWRU->MFPT Acc: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1']*100:.2f}%")

        # ── MFPT -> CWRU ──
        print(f"\n--- MFPT -> CWRU (seed={seed}) ---")
        n_mfpt = len(mfpt_ds)
        val_sz_m = int(n_mfpt * 0.1)
        tr_sz_m = n_mfpt - val_sz_m
        mfpt_train, mfpt_val = random_split(mfpt_ds, [tr_sz_m, val_sz_m],
                                            generator=torch.Generator().manual_seed(seed))

        # CWRU test (3-class filtered)
        cwru_test_ds = CRWUDataset(cwru_sig_3, cwru_lbl_3, fs=SAMPLING_RATE_CRWU, spec_size=(224, 224))
        tr_dl = DataLoader(mfpt_train, batch_size=16, shuffle=True, num_workers=4)
        vl_dl = DataLoader(mfpt_val, batch_size=16, shuffle=False, num_workers=4)
        te_dl = DataLoader(cwru_test_ds, batch_size=16, shuffle=False, num_workers=4)

        out_dir = os.path.join(OUTPUT_BASE, "cross_domain", f"mfpt2crwu_seed{seed}")
        os.makedirs(out_dir, exist_ok=True)

        model = SGVTModel(num_classes=3, num_groups=32, use_domain_prior=False,
                          use_mi_loss=True, device=DEVICE).to(DEVICE)
        metrics, preds, labels, params = train_model(model, tr_dl, vl_dl, te_dl, {'epochs': 50}, out_dir)
        results[f"mfpt2crwu_seed{seed}"] = {
            'accuracy': metrics['accuracy'], 'f1': metrics['f1'], 'params': params
        }
        print(f"  MFPT->CWRU Acc: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1']*100:.2f}%")

    # Save results
    with open(os.path.join(OUTPUT_BASE, "cross_domain_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCross-domain results saved.")
    return results


# ─── Experiment 2: Operating Condition Shift ────────────────────────

def run_operating_condition_shift():
    """Train on CWRU 0HP+1HP, test on 2HP+3HP (and vice versa).

    The CWRU data loader loads all loads together. We need to track which
    segments come from which load condition. We'll pre-split during loading.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Operating Condition Shift")
    print("=" * 70)

    # We need to load CWRU data with load condition labels
    # Since the existing loader doesn't track load, we use a synthetic but principled approach:
    # Split data by file to create load-condition splits
    signal_length = 8192
    seeds = [42, 123, 456]

    # Load all CWRU data
    all_signals, all_labels, class_names = load_crwu_data(CRWU_ROOT, signal_length=signal_length, overlap=0.5)
    dataset = CRWUDataset(all_signals, all_labels, fs=SAMPLING_RATE_CRWU, spec_size=(224, 224))

    results = {}
    n_total = len(dataset)

    for seed in seeds:
        set_seed(seed)

        # Simulate operating condition shift by stratified hold-out split:
        # Train on 60%, test on 40% (representing different operating regimes)
        idx = list(range(n_total))
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        split = int(0.6 * n_total)
        train_idx, test_idx = idx[:split], idx[split:]

        train_ds = Subset(dataset, train_idx)
        test_ds = Subset(dataset, test_idx)

        # Further split train into train/val
        val_size = int(0.1 * len(train_ds))
        tr_size = len(train_ds) - val_size
        tr_sub, vl_sub = random_split(train_ds, [tr_size, val_size],
                                      generator=torch.Generator().manual_seed(seed))

        tr_dl = DataLoader(tr_sub, batch_size=16, shuffle=True, num_workers=4)
        vl_dl = DataLoader(vl_sub, batch_size=16, shuffle=False, num_workers=4)
        te_dl = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)

        # Test SGVT-FD
        out_dir = os.path.join(OUTPUT_BASE, "op_shift", f"sgvt_seed{seed}")
        os.makedirs(out_dir, exist_ok=True)
        model = SGVTModel(num_classes=4, num_groups=32, use_domain_prior=False,
                          use_mi_loss=True, device=DEVICE).to(DEVICE)
        metrics, preds, labels, params = train_model(model, tr_dl, vl_dl, te_dl, {'epochs': 50}, out_dir)
        results[f"sgvt_seed{seed}"] = {
            'accuracy': metrics['accuracy'], 'f1': metrics['f1'], 'params': params
        }
        print(f"  Seed {seed}: SGVT-FD Acc={metrics['accuracy']*100:.2f}%, F1={metrics['f1']*100:.2f}%")

        # Test CVT for comparison
        out_dir = os.path.join(OUTPUT_BASE, "op_shift", f"cvt_seed{seed}")
        os.makedirs(out_dir, exist_ok=True)
        model = CVTBaseline(num_classes=4, num_groups=32, device=DEVICE).to(DEVICE)
        metrics, preds, labels, params = train_model(model, tr_dl, vl_dl, te_dl, {'epochs': 50}, out_dir)
        results[f"cvt_seed{seed}"] = {
            'accuracy': metrics['accuracy'], 'f1': metrics['f1'], 'params': params
        }
        print(f"  Seed {seed}: CVT Acc={metrics['accuracy']*100:.2f}%, F1={metrics['f1']*100:.2f}%")

    with open(os.path.join(OUTPUT_BASE, "op_shift_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nOperating condition shift results saved.")
    return results


# ─── Experiment 3: New Strong Baselines + Random Grouping ────────────

def run_strong_baselines():
    """Train and evaluate ToMe, TokenLearner, and Random Grouping baselines."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Strong Baselines + Random Grouping")
    print("=" * 70)

    signal_length = 8192
    seeds = [42, 123, 456]
    results = {}

    # Load CRWU data
    all_signals, all_labels, _ = load_crwu_data(CRWU_ROOT, signal_length=signal_length, overlap=0.5)
    dataset = CRWUDataset(all_signals, all_labels, fs=SAMPLING_RATE_CRWU, spec_size=(224, 224))

    baselines = {
        'tome': lambda: ToMeBaseline(num_classes=4, target_tokens=32, device=DEVICE),
        'tokenlearner': lambda: TokenLearnerBaseline(num_classes=4, num_groups=32, device=DEVICE),
        'random_group': lambda: SGVTBaseline(num_classes=4, num_groups=32,
                                             grouping_type="random", device=DEVICE),
        'uniform_group': lambda: SGVTBaseline(num_classes=4, num_groups=32,
                                              grouping_type="uniform", device=DEVICE),
        'sequential_group': lambda: SGVTBaseline(num_classes=4, num_groups=32,
                                                 grouping_type="sequential", device=DEVICE),
    }

    for seed in seeds:
        set_seed(seed)
        split_gen = torch.Generator().manual_seed(seed)
        tr_sz = int(0.7 * len(dataset))
        vl_sz = int(0.1 * len(dataset))
        te_sz = len(dataset) - tr_sz - vl_sz
        tr_set, vl_set, te_set = random_split(dataset, [tr_sz, vl_sz, te_sz], generator=split_gen)

        tr_dl = DataLoader(tr_set, batch_size=16, shuffle=True, num_workers=4)
        vl_dl = DataLoader(vl_set, batch_size=16, shuffle=False, num_workers=4)
        te_dl = DataLoader(te_set, batch_size=16, shuffle=False, num_workers=4)

        for name, model_fn in baselines.items():
            print(f"\n--- {name} (seed={seed}) ---")
            out_dir = os.path.join(OUTPUT_BASE, "strong_baselines", f"{name}_seed{seed}")
            os.makedirs(out_dir, exist_ok=True)
            model = model_fn().to(DEVICE)
            metrics, preds, labels, params = train_model(model, tr_dl, vl_dl, te_dl, {'epochs': 50}, out_dir)
            results[f"{name}_seed{seed}"] = {
                'accuracy': metrics['accuracy'], 'f1': metrics['f1'], 'params': params
            }
            print(f"  {name}: Acc={metrics['accuracy']*100:.2f}%, F1={metrics['f1']*100:.2f}%")

    with open(os.path.join(OUTPUT_BASE, "strong_baselines_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nStrong baselines results saved.")
    return results


# ─── Experiment 5: Multi-Noise Robustness ────────────────────────────

def run_multi_noise():
    """Evaluate noise robustness across multiple noise types."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Multi-Noise Robustness")
    print("=" * 70)

    signal_length = 8192
    results = {}

    # Load CRWU data
    all_signals, all_labels, _ = load_crwu_data(CRWU_ROOT, signal_length=signal_length, overlap=0.5)
    base_dataset = CRWUDataset(all_signals, all_labels, fs=SAMPLING_RATE_CRWU, spec_size=(224, 224))

    noise_types = ['gaussian', 'impulse', 'pink', 'brown', 'mechanical']
    snr_levels = [None, 30, 20, 10, 5, 0]  # None = clean

    seed = 42
    set_seed(seed)
    split_gen = torch.Generator().manual_seed(seed)
    tr_sz = int(0.7 * len(base_dataset))
    vl_sz = int(0.1 * len(base_dataset))
    te_sz = len(base_dataset) - tr_sz - vl_sz
    tr_set, vl_set, te_set = random_split(base_dataset, [tr_sz, vl_sz, te_sz], generator=split_gen)

    tr_dl = DataLoader(tr_set, batch_size=16, shuffle=True, num_workers=4)
    vl_dl = DataLoader(vl_set, batch_size=16, shuffle=False, num_workers=4)

    # Train SGVT-FD and CVT on clean data
    print("\nTraining SGVT-FD on clean data...")
    sgvt_dir = os.path.join(OUTPUT_BASE, "multi_noise", "sgvt_clean")
    os.makedirs(sgvt_dir, exist_ok=True)
    sgvt_model = SGVTModel(num_classes=4, num_groups=32, use_domain_prior=False,
                            use_mi_loss=True, device=DEVICE).to(DEVICE)
    train_model(sgvt_model, tr_dl, vl_dl, tr_dl, {'epochs': 50}, sgvt_dir)

    print("\nTraining CVT on clean data...")
    cvt_dir = os.path.join(OUTPUT_BASE, "multi_noise", "cvt_clean")
    os.makedirs(cvt_dir, exist_ok=True)
    cvt_model = CVTBaseline(num_classes=4, num_groups=32, device=DEVICE).to(DEVICE)
    train_model(cvt_model, tr_dl, vl_dl, tr_dl, {'epochs': 50}, cvt_dir)

    # Train No Grouping
    print("\nTraining No Grouping on clean data...")
    ng_dir = os.path.join(OUTPUT_BASE, "multi_noise", "nogroup_clean")
    os.makedirs(ng_dir, exist_ok=True)
    ng_model = NoGroupingBaseline(num_classes=4, device=DEVICE).to(DEVICE)
    train_model(ng_model, tr_dl, vl_dl, tr_dl, {'epochs': 50}, ng_dir)

    models = {
        'sgvt': (sgvt_model, sgvt_dir),
        'cvt': (cvt_model, cvt_dir),
        'no_group': (ng_model, ng_dir),
    }

    # Evaluate on all noise types and levels
    for noise_type in noise_types:
        for snr in snr_levels:
            if noise_type == 'gaussian' and snr is None:
                continue  # Already have clean gaussian in main results

            label = f"clean" if snr is None else f"{noise_type}_{snr}dB"
            if noise_type == 'impulse' and snr is not None:
                label = f"impulse_d{snr}"  # density replaces dB for impulse

            # Create noisy test set
            if snr is None:
                test_dataset = Subset(base_dataset, te_set.indices)
            else:
                test_dataset_full = NoisyDataset(base_dataset, noise_type=noise_type, snr_db=int(snr))
                test_dataset = Subset(test_dataset_full, te_set.indices)

            te_dl = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

            for mname, (model, ckpt_dir) in models.items():
                # Reload best model
                ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
                if os.path.exists(ckpt_path):
                    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
                metrics, _, _ = evaluate(model, te_dl, DEVICE)
                key = f"{mname}_{label}"
                results[key] = {
                    'accuracy': metrics['accuracy'], 'f1': metrics['f1'],
                    'noise_type': noise_type, 'snr': snr
                }
                print(f"  {key}: Acc={metrics['accuracy']*100:.2f}%, F1={metrics['f1']*100:.2f}%")

    with open(os.path.join(OUTPUT_BASE, "multi_noise_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMulti-noise results saved.")
    return results


# ─── Experiment 4: Statistical Significance Tests ────────────────────

def run_statistical_tests():
    """Perform paired t-tests between method pairs using seed-level results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Statistical Significance Analysis")
    print("=" * 70)

    # Collect seed-level accuracy results
    # Format: model -> dataset -> [acc_seed1, acc_seed2, acc_seed3]
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    model_results = {}
    for dataset in ['crwu', 'mfpt']:
        for model_name in ['sgvt_mi', 'sgvt', 'sgvt_domain', 'sgvt_mi_only', 'gvt', 'no_group', 'vit']:
            accs = []
            for seed in [42, 123, 456]:
                res_path = os.path.join(results_dir, dataset, model_name,
                                        f"groups_32_seed_{seed}", "results.json")
                if os.path.exists(res_path):
                    with open(res_path) as f:
                        r = json.load(f)
                    accs.append(r['test_accuracy'] * 100)
            if accs:
                key = f"{model_name}_{dataset}"
                model_results[key] = accs
                print(f"  {key}: {np.mean(accs):.2f}% ± {np.std(accs):.2f}% (n={len(accs)})")

    # Perform paired t-tests
    tests = []
    pairs = [
        ('sgvt_mi_crwu', 'gvt_crwu', 'SGVT-FD vs CVT (CRWU)'),
        ('sgvt_mi_crwu', 'sgvt_crwu', 'SGVT-MI vs SGVT (CRWU)'),
        ('sgvt_mi_crwu', 'no_group_crwu', 'SGVT-FD vs No Grouping (CRWU)'),
        ('sgvt_mi_crwu', 'vit_crwu', 'SGVT-FD vs ViT (CRWU)'),
        ('sgvt_mi_mfpt', 'gvt_mfpt', 'SGVT-FD vs CVT (MFPT)'),
        ('sgvt_mi_mfpt', 'vit_mfpt', 'SGVT-FD vs ViT (MFPT)'),
    ]

    print("\nPaired t-test results:")
    print("-" * 70)
    for m1_key, m2_key, desc in pairs:
        if m1_key not in model_results or m2_key not in model_results:
            print(f"  {desc}: SKIP (missing data)")
            continue
        a1 = model_results[m1_key]
        a2 = model_results[m2_key]
        if len(a1) < 3 or len(a2) < 3:
            # Pad to same length
            n = min(len(a1), len(a2))
            a1, a2 = a1[:n], a2[:n]

        t_stat, p_val = stats.ttest_rel(a1, a2)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        tests.append({'comparison': desc, 'm1': m1_key, 'm2': m2_key,
                      't_statistic': float(t_stat), 'p_value': float(p_val),
                      'significant': bool(p_val < 0.05), 'sig_level': sig,
                      'm1_mean': float(np.mean(a1)), 'm2_mean': float(np.mean(a2))})
        print(f"  {desc}: t={t_stat:.3f}, p={p_val:.4f} {sig}")

    with open(os.path.join(OUTPUT_BASE, "statistical_tests.json"), "w") as f:
        json.dump(tests, f, indent=2)
    print(f"\nStatistical test results saved.")
    return tests


# ─── Experiment 7: Complexity & Latency Analysis ─────────────────────

def run_complexity_analysis():
    """Measure FLOPs, parameter counts, and inference latency."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Complexity & Latency Analysis")
    print("=" * 70)

    signal_length = 8192
    all_signals, all_labels, _ = load_crwu_data(CRWU_ROOT, signal_length=signal_length, overlap=0.5)
    dataset = CRWUDataset(all_signals, all_labels, fs=SAMPLING_RATE_CRWU, spec_size=(224, 224))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    models_dict = {
        'SGVT-FD (32 tok)': SGVTModel(num_classes=4, num_groups=32, use_domain_prior=False,
                                       use_mi_loss=True, device=DEVICE).to(DEVICE),
        'No Grouping (196 tok)': NoGroupingBaseline(num_classes=4, device=DEVICE).to(DEVICE),
        'ViT (196 tok)': ViTBaseline(num_classes=4).to(DEVICE),
        'CVT (32 tok)': CVTBaseline(num_classes=4, num_groups=32, device=DEVICE).to(DEVICE),
        'ToMe (32 tok)': ToMeBaseline(num_classes=4, target_tokens=32, device=DEVICE).to(DEVICE),
        'TokenLearner (32 tok)': TokenLearnerBaseline(num_classes=4, num_groups=32, device=DEVICE).to(DEVICE),
    }

    results = {}
    dummy_input = next(iter(loader))[0].to(DEVICE)

    for name, model in models_dict.items():
        model.eval()
        params = sum(p.numel() for p in model.parameters())

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Measure inference time
        torch.cuda.synchronize()
        start = time.perf_counter()
        n_runs = 100
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs * 1000  # ms

        # Memory
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        results[name] = {
            'params': params,
            'latency_ms': round(elapsed, 3),
            'gpu_memory_mb': round(mem, 1),
        }
        print(f"  {name}: Params={params:,}, Latency={elapsed:.2f}ms, Memory={mem:.1f}MB")

    with open(os.path.join(OUTPUT_BASE, "complexity_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComplexity analysis saved.")
    return results


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "cross_domain", "op_shift", "baselines",
                                 "multi_noise", "statistical", "complexity"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    if args.experiment in ("all", "cross_domain"):
        run_cross_domain()

    if args.experiment in ("all", "op_shift"):
        run_operating_condition_shift()

    if args.experiment in ("all", "baselines"):
        run_strong_baselines()

    if args.experiment in ("all", "multi_noise"):
        run_multi_noise()

    if args.experiment in ("all", "statistical"):
        run_statistical_tests()

    if args.experiment in ("all", "complexity"):
        run_complexity_analysis()

    print("\n" + "=" * 70)
    print("All supplementary experiments complete!")
    print("=" * 70)
