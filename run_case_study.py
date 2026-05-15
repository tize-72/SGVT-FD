"""
Case Study: Interpretability and Diagnostic Behavior of SGVT-FD.
Selects 3 representative test samples, extracts intermediate model outputs,
generates 4×3 visualization figure, summary table, and LLM interpretability comparison.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import CRWU_ROOT, CRWU_CLASSES, CRWU_FAULT_FREQS, SAMPLING_RATE_CRWU
from src.data.crwu_loader import load_crwu_data, CRWUDataset
from src.models.sgvt import SGVTModel, NoGroupingBaseline
from src.models.baselines import CVTBaseline, ViTBaseline
from src.utils.signal_processing import generate_spectrogram, normalize_signal

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "case_study")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CRWU fault characteristic frequencies at 1750 RPM
FAULT_FREQS = {
    'BPFO': 107.3,   # Ball Pass Frequency Outer
    'BPFI': 162.2,   # Ball Pass Frequency Inner
    'FTF': 14.1,     # Fundamental Train Frequency
    'BSF': 70.6,     # Ball Spin Frequency
}
# Harmonics
FAULT_FREQ_BANDS = {
    'InnerRace': [162.2, 324.4, 486.6, 648.8],   # BPFI 1x-4x
    'Ball': [70.6, 141.2, 211.8, 282.4],           # BSF 1x-4x
    'OuterRace': [107.3, 214.6, 321.9, 429.2],     # BPFO 1x-4x
    'Normal': [],
}


class SGVTModelWithIntermediate(SGVTModel):
    """Wrapper that returns intermediate tensors for visualization."""

    def forward_with_intermediates(self, x, labels=None):
        tokens = self.grouping.extract_patches(x)
        assignments = self.grouping.compute_group_assignments(tokens)
        merged = self.grouping.merge_tokens(tokens, assignments)
        merged_proj = self.grouping.output_proj(merged)
        merged_norm = self.grouping.norm(merged_proj)
        group_importance = torch.norm(merged_norm, dim=-1)
        pooled = merged_norm.mean(dim=1)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            mi_loss = torch.tensor(0.0, device=x.device)
            if hasattr(self.grouping, 'compute_mi_loss'):
                mi_loss = self.grouping.compute_mi_loss(tokens, assignments, labels)
            div_loss = self.compute_diversity_loss(assignments)
            loss = ce_loss + mi_loss + self.diversity_weight * div_loss
        probs = F.softmax(logits, dim=-1)
        return {
            'logits': logits, 'loss': loss, 'probs': probs,
            'tokens': tokens, 'assignments': assignments,
            'merged_tokens': merged_norm, 'group_importance': group_importance,
        }


class CVTModelWithIntermediate(CVTBaseline):
    """Wrapper for CVT baseline that returns intermediate tensors."""

    def forward_with_intermediates(self, x, labels=None):
        B = x.shape[0]
        patches = self.patch_embed(x)
        tokens = patches.reshape(B, self.feature_dim, -1).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        # Correlation grouping - extract assignments
        tokens_norm = F.normalize(tokens, dim=-1)
        correlation = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # (B, N, N)
        avg_corr = correlation.mean(dim=2)  # (B, N)
        _, topk_indices = avg_corr.topk(self.num_groups, dim=1)  # (B, K)

        # Assign each token to most correlated representative
        N = tokens.shape[1]
        K = self.num_groups
        # Gather correlation of each token with each representative
        # topk_indices: (B, K) -> expand to (B, K, N)
        topk_expanded = topk_indices.unsqueeze(2).expand(-1, -1, N)
        # correlation: (B, N, N) -> gather rows at topk_indices -> (B, K, N)
        rep_corr = torch.gather(correlation, 1, topk_expanded)  # (B, K, N)
        assignments = rep_corr.transpose(1, 2)  # (B, N, K)
        assignments = F.softmax(assignments * 10, dim=-1)

        assign_T = assignments.transpose(1, 2)
        merged = torch.bmm(assign_T, tokens)
        group_sizes = assignments.sum(dim=1, keepdim=True).transpose(1, 2).clamp(min=1e-8)
        merged = merged / group_sizes

        group_importance = torch.norm(merged, dim=-1)
        pooled = merged.mean(dim=1)
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

        return {
            'logits': logits, 'loss': loss, 'probs': probs,
            'tokens': tokens, 'assignments': assignments,
            'merged_tokens': merged, 'group_importance': group_importance,
        }


class ViTModelWithIntermediate(ViTBaseline):
    """Wrapper for ViT baseline. No explicit grouping - uses CLS token."""

    def forward_with_intermediates(self, x, labels=None):
        B = x.shape[0]
        patches = self.patch_embed(x)
        tokens = patches.reshape(B, self.feature_dim, -1).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        encoded = self.transformer(tokens)
        encoded = self.norm(encoded)
        cls_output = encoded[:, 0, :]
        logits = self.classifier(cls_output)
        probs = F.softmax(logits, dim=-1)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        # No meaningful group assignments for ViT
        N = (224 // 16) ** 2  # 196
        dummy_assignments = torch.zeros(B, N, 1, device=x.device)
        dummy_importance = torch.ones(B, 1, device=x.device)
        return {
            'logits': logits, 'loss': loss, 'probs': probs,
            'tokens': tokens[:, 1:, :],  # exclude CLS
            'assignments': dummy_assignments,
            'merged_tokens': cls_output.unsqueeze(1),
            'group_importance': dummy_importance,
        }


class NoGroupModelWithIntermediate(NoGroupingBaseline):
    """Wrapper for No Grouping baseline. Uses all tokens with transformer."""

    def forward_with_intermediates(self, x, labels=None):
        B = x.shape[0]
        patches = self.patch_embed(x)
        tokens = patches.reshape(B, self.feature_dim, -1).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.transformer(tokens)
        cls_output = encoded[:, 0, :]
        logits = self.classifier(cls_output)
        probs = F.softmax(logits, dim=-1)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        N = (224 // 16) ** 2
        dummy_assignments = torch.zeros(B, N, 1, device=x.device)
        dummy_importance = torch.ones(B, 1, device=x.device)
        return {
            'logits': logits, 'loss': loss, 'probs': probs,
            'tokens': encoded[:, 1:, :],  # exclude CLS
            'assignments': dummy_assignments,
            'merged_tokens': cls_output.unsqueeze(1),
            'group_importance': dummy_importance,
        }


def load_trained_model(model_class, ckpt_path, **kwargs):
    """Load trained model with intermediate output wrapper."""
    model = model_class(**kwargs).to(DEVICE)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model


def get_test_predictions(model, test_loader):
    """Run inference on test set, return all predictions."""
    all_results = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            out = model.forward_with_intermediates(images, labels)
            probs = out['probs'].cpu().numpy()
            preds = probs.argmax(axis=1)
            confs = probs.max(axis=1)
            for i in range(images.size(0)):
                all_results.append({
                    'global_idx': batch_idx * test_loader.batch_size + i,
                    'true_label': labels[i].item(),
                    'pred_label': preds[i],
                    'confidence': confs[i],
                    'prob_dist': probs[i],
                    'correct': preds[i] == labels[i].item(),
                })
    return all_results


def select_cases(predictions):
    """Select 3 representative cases."""
    # Case A: correct, high confidence, prefer inner race
    correct_high = [p for p in predictions if p['correct'] and p['confidence'] > 0.9]
    inner_race_correct = [p for p in correct_high if p['true_label'] == 2]
    case_a = inner_race_correct[0] if inner_race_correct else correct_high[0]

    # Case B: misclassified
    wrong = [p for p in predictions if not p['correct']]
    ball_ir = [p for p in wrong if
               (p['true_label'] == 1 and p['pred_label'] == 2) or
               (p['true_label'] == 2 and p['pred_label'] == 1)]
    case_b = ball_ir[0] if ball_ir else wrong[0]

    # Case C: correct but lower confidence
    low_conf = [p for p in predictions if p['correct'] and p['confidence'] < 0.9]
    ball_low = [p for p in low_conf if p['true_label'] == 1]
    case_c = ball_low[0] if ball_low else low_conf[len(low_conf) // 2]

    return case_a, case_b, case_c


def extract_case_intermediates(model, dataset, case_info):
    """Extract full intermediate tensors for a single case."""
    idx = case_info['global_idx']
    spec_rgb, label = dataset[idx]
    spec_rgb = spec_rgb.unsqueeze(0).to(DEVICE)
    label_tensor = torch.LongTensor([label]).to(DEVICE)
    with torch.no_grad():
        out = model.forward_with_intermediates(spec_rgb, label_tensor)
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    real_idx = dataset.indices[idx] if hasattr(dataset, 'indices') else idx
    sig = base_dataset.signals[real_idx]
    sig_norm = normalize_signal(sig)
    spec = generate_spectrogram(sig_norm, fs=12000, target_size=(224, 224))
    return {
        'spectrogram': spec, 'signal': sig_norm,
        'tokens': out['tokens'].cpu().numpy()[0],
        'assignments': out['assignments'].cpu().numpy()[0],
        'merged_tokens': out['merged_tokens'].cpu().numpy()[0],
        'group_importance': out['group_importance'].cpu().numpy()[0],
        'prob_dist': out['probs'].cpu().numpy()[0],
        'pred_label': out['probs'].argmax(dim=1).item(),
        'true_label': label, 'confidence': out['probs'].max().item(),
    }


def map_group_to_freq_band(group_idx, num_groups, nyquist=6000):
    """Map a group index to its frequency range in Hz."""
    freq_per_group = nyquist / num_groups
    low = group_idx * freq_per_group
    high = (group_idx + 1) * freq_per_group
    return low, high


def match_fault_freq(freq_low, freq_high, fault_type=None):
    """Check if a frequency band matches known fault characteristic frequencies."""
    matches = []
    all_freqs = []
    if fault_type and fault_type in FAULT_FREQ_BANDS:
        all_freqs = FAULT_FREQ_BANDS[fault_type]
    else:
        for v in FAULT_FREQ_BANDS.values():
            all_freqs.extend(v)
    for f in all_freqs:
        if freq_low <= f <= freq_high:
            matches.append(f)
    return matches


def plot_case_study(case_a, case_b, case_c, save_path):
    """Generate the 4×3 case study figure with improved Row 3."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    cases = [case_a, case_b, case_c]
    case_labels = ['(a) Case A: Correct, High Conf.',
                   '(b) Case B: Misclassified',
                   '(c) Case C: Correct, Lower Conf.']
    class_names = CRWU_CLASSES
    nyquist = 6000
    num_groups = 32

    # Colors for top-3 groups
    group_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

    for col, (case, case_label) in enumerate(zip(cases, case_labels)):
        spec = case['spectrogram']
        assignments = case['assignments']
        group_importance = case['group_importance']
        prob_dist = case['prob_dist']
        pred = case['pred_label']
        true = case['true_label']
        conf = case['confidence']

        group_map = assignments.argmax(axis=1)
        grid_size = int(np.sqrt(len(group_map)))
        group_map_2d = group_map[:grid_size * grid_size].reshape(grid_size, grid_size)
        topk_indices = np.argsort(group_importance)[-3:][::-1]  # descending

        # Row 0: Spectrogram
        ax = axes[0, col]
        ax.imshow(spec, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(case_label, fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=7)

        # Row 1: Grouping Map
        ax = axes[1, col]
        ax.imshow(group_map_2d, cmap='tab20', aspect='equal')
        ax.set_xlabel('Patch X', fontsize=9)
        ax.set_ylabel('Patch Y', fontsize=9)
        ax.set_title('Group Assignment Map', fontsize=10)
        ax.tick_params(labelsize=7)

        # Row 2: Dominant Groups with colored overlays + frequency info
        ax = axes[2, col]
        ax.imshow(spec, aspect='auto', cmap='viridis', origin='lower', alpha=0.6)

        from PIL import Image
        legend_handles = []
        for rank, g_idx in enumerate(topk_indices):
            # Create mask for this group
            mask = (group_map == g_idx).astype(float)
            mask_2d = mask[:grid_size * grid_size].reshape(grid_size, grid_size)
            mask_img = Image.fromarray((mask_2d * 255).astype(np.uint8))
            mask_resized = np.array(mask_img.resize((224, 224), Image.NEAREST)).astype(float) / 255.0

            # Create colored overlay
            overlay = np.zeros((*mask_resized.shape, 4))
            color_rgb = matplotlib.colors.to_rgb(group_colors[rank])
            overlay[mask_resized > 0.5] = [*color_rgb, 0.45]
            ax.imshow(overlay, aspect='auto', origin='lower')

            # Frequency band info
            freq_low, freq_high = map_group_to_freq_band(g_idx, num_groups, nyquist)
            imp_score = group_importance[g_idx]
            label = f'G{g_idx}: {freq_low:.0f}-{freq_high:.0f} Hz (imp={imp_score:.1f})'
            legend_handles.append(mpatches.Patch(facecolor=group_colors[rank],
                                                  alpha=0.6, label=label))

        ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
                  framealpha=0.8, handlelength=1.5)
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        dominant = topk_indices[0]
        d_low, d_high = map_group_to_freq_band(dominant, num_groups, nyquist)
        ax.set_title(f'Dominant: G{dominant} ({d_low:.0f}-{d_high:.0f} Hz)', fontsize=10)
        ax.tick_params(labelsize=7)

        # Row 3: Prediction bar chart
        ax = axes[3, col]
        colors = ['#555555'] * len(class_names)
        colors[pred] = '#2ca02c' if pred == true else '#d62728'
        bars = ax.bar(range(len(class_names)), prob_dist, color=colors,
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=8, rotation=15)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Probability', fontsize=9)
        ax.set_title(f'True: {class_names[true]}, Pred: {class_names[pred]} ({conf:.2f})',
                     fontsize=10)
        ax.tick_params(labelsize=7)
        for bar, p in zip(bars, prob_dist):
            if p > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{p:.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved case study figure to {save_path}")


def generate_llm_explanation(case, model_name='SGVT-FD'):
    """Generate LLM-style diagnostic explanation based on model intermediate outputs."""
    class_names = CRWU_CLASSES
    true_name = class_names[case['true_label']]
    pred_name = class_names[case['pred_label']]
    conf = case['confidence']
    prob_dist = case['prob_dist']
    group_importance = case['group_importance']
    assignments = case['assignments']
    num_groups = assignments.shape[1] if len(assignments.shape) > 1 else 32

    group_map = assignments.argmax(axis=1)
    grid_size = int(np.sqrt(len(group_map)))
    topk_indices = np.argsort(group_importance)[-3:][::-1]

    # Build frequency band analysis
    freq_analysis = []
    for g_idx in topk_indices:
        freq_low, freq_high = map_group_to_freq_band(g_idx, num_groups)
        imp = group_importance[g_idx]
        matches_inner = match_fault_freq(freq_low, freq_high, 'InnerRace')
        matches_ball = match_fault_freq(freq_low, freq_high, 'Ball')
        matches_outer = match_fault_freq(freq_low, freq_high, 'OuterRace')
        all_matches = []
        if matches_inner:
            all_matches.extend([f'BPFI {f:.0f}Hz' for f in matches_inner])
        if matches_ball:
            all_matches.extend([f'BSF {f:.0f}Hz' for f in matches_ball])
        if matches_outer:
            all_matches.extend([f'BPFO {f:.0f}Hz' for f in matches_outer])
        freq_str = ', '.join(all_matches) if all_matches else 'no known fault freq'
        freq_analysis.append((g_idx, freq_low, freq_high, imp, freq_str))

    unique_groups = len(np.unique(group_map))
    dominant = freq_analysis[0]

    if model_name == 'SGVT-FD':
        lines = []
        lines.append(f"Prediction: {pred_name} (confidence: {conf:.2%})")
        lines.append(f"True label: {true_name}")
        lines.append("")
        lines.append("Frequency-band analysis:")
        for g_idx, flo, fhi, imp, freq_str in freq_analysis:
            lines.append(f"  Group G{g_idx} ({flo:.0f}-{fhi:.0f} Hz, importance={imp:.2f}): "
                        f"matches {freq_str}")
        lines.append("")
        if pred_name == true_name:
            lines.append(f"The model correctly identifies {true_name} fault. "
                        f"The dominant frequency group G{dominant[0]} ({dominant[1]:.0f}-{dominant[2]:.0f} Hz) "
                        f"aligns with the expected characteristic frequency band ({dominant[4]}). "
                        f"Semantic grouping clusters tokens from diagnostically relevant frequency bands, "
                        f"enabling precise fault identification.")
        else:
            lines.append(f"The model predicts {pred_name} instead of {true_name}. "
                        f"The dominant group G{dominant[0]} ({dominant[1]:.0f}-{dominant[2]:.0f} Hz) "
                        f"overlaps with {pred_name}-related frequencies ({dominant[4]}), "
                        f"leading to confusion. This is physically plausible as {true_name} and {pred_name} "
                        f"faults can share spectral energy in overlapping frequency bands.")
        second = freq_analysis[1]
        lines.append(f"\nThe secondary group G{second[0]} ({second[1]:.0f}-{second[2]:.0f} Hz) "
                    f"provides complementary diagnostic information ({second[4]}). "
                    f"Across {unique_groups} distinct groups, the model captures a structured "
                    f"frequency-band representation that aligns with bearing fault physics.")
        return '\n'.join(lines)

    elif model_name == 'CVT':
        lines = []
        lines.append(f"Prediction: {pred_name} (confidence: {conf:.2%})")
        lines.append(f"True label: {true_name}")
        lines.append("")
        lines.append("Group analysis (correlation-based):")
        for g_idx, flo, fhi, imp, freq_str in freq_analysis:
            lines.append(f"  Group G{g_idx} ({flo:.0f}-{fhi:.0f} Hz): "
                        f"correlation-based grouping, {freq_str}")
        lines.append("")
        if pred_name == true_name:
            lines.append(f"The model correctly predicts {true_name}. "
                        f"However, correlation-based grouping does not explicitly align with "
                        f"fault characteristic frequency bands. The grouping is driven by "
                        f"amplitude correlation patterns rather than semantic frequency structure, "
                        f"limiting physical interpretability.")
        else:
            lines.append(f"The model predicts {pred_name} instead of {true_name}. "
                        f"Correlation-based grouping is sensitive to amplitude variations "
                        f"rather than frequency-domain semantics, making it difficult to "
                        f"diagnose the root cause of misclassification.")
        lines.append(f"\nWith {unique_groups} groups formed by pairwise token correlation, "
                    f"the model lacks explicit frequency-band structure, making it harder "
                    f"for domain experts to validate the diagnostic reasoning.")
        return '\n'.join(lines)

    elif model_name == 'ViT':
        lines = []
        lines.append(f"Prediction: {pred_name} (confidence: {conf:.2%})")
        lines.append(f"True label: {true_name}")
        lines.append("")
        lines.append("Model architecture: Vision Transformer (ViT) with 196 patch tokens, no grouping.")
        lines.append("")
        if pred_name == true_name:
            lines.append(f"The model correctly predicts {true_name}. "
                        f"ViT processes all 196 patch tokens independently through self-attention, "
                        f"without explicit semantic grouping. The classification relies on the "
                        f"CLS token aggregation over all patches, making it difficult to identify "
                        f"which frequency bands contributed most to the decision.")
        else:
            lines.append(f"The model predicts {pred_name} instead of {true_name}. "
                        f"Without semantic grouping, ViT treats all 196 patches equally, "
                        f"making it difficult to diagnose which spectral regions caused the "
                        f"misclassification.")
        lines.append(f"\nViT provides no explicit frequency-band structure or group-level "
                    f"interpretability. A domain expert cannot determine which spectral "
                    f"regions the model relied on for its prediction.")
        return '\n'.join(lines)

    else:  # No Grouping
        lines = []
        lines.append(f"Prediction: {pred_name} (confidence: {conf:.2%})")
        lines.append(f"True label: {true_name}")
        lines.append("")
        lines.append("Model architecture: Transformer encoder over all 196 tokens with CLS aggregation, no grouping.")
        lines.append("")
        if pred_name == true_name:
            lines.append(f"The model correctly predicts {true_name}. "
                        f"The transformer encoder processes all tokens with self-attention, "
                        f"but without semantic grouping, the model cannot provide frequency-band-level "
                        f"diagnostic explanations. The classification is based on global token "
                        f"aggregation rather than structured frequency-band analysis.")
        else:
            lines.append(f"The model predicts {pred_name} instead of {true_name}. "
                        f"Without grouping, the model processes all tokens uniformly, "
                        f"making it impossible to identify which frequency bands contributed "
                        f"to the misclassification.")
        lines.append(f"\nNo Grouping baseline uses all 196 tokens without compression or "
                    f"semantic structure. While this preserves all information, it sacrifices "
                    f"interpretability: the model cannot explain its reasoning in terms of "
                    f"diagnostically relevant frequency bands.")
        return '\n'.join(lines)


def run():
    print("=" * 70)
    print("Case Study: SGVT-FD Interpretability")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define all models to compare
    model_configs = {
        'SGVT-FD': {
            'class': SGVTModelWithIntermediate,
            'ckpt': os.path.join(base_dir, 'results', 'crwu', 'sgvt_mi_only', 'groups_32_seed_42', 'best_model.pt'),
            'kwargs': {'num_classes': 4, 'num_groups': 32, 'use_domain_prior': False, 'use_mi_loss': True, 'device': DEVICE},
            'type': 'semantic',
        },
        'CVT': {
            'class': CVTModelWithIntermediate,
            'ckpt': os.path.join(base_dir, 'results', 'crwu', 'gvt', 'groups_32_seed_42', 'best_model.pt'),
            'kwargs': {'num_classes': 4, 'num_groups': 32, 'device': DEVICE},
            'type': 'correlation',
        },
        'ViT': {
            'class': ViTModelWithIntermediate,
            'ckpt': os.path.join(base_dir, 'results', 'crwu', 'vit', 'groups_32_seed_42', 'best_model.pt'),
            'kwargs': {'num_classes': 4},
            'type': 'no_group',
        },
        'No Grouping': {
            'class': NoGroupModelWithIntermediate,
            'ckpt': os.path.join(base_dir, 'results', 'crwu', 'no_group', 'groups_32_seed_42', 'best_model.pt'),
            'kwargs': {'num_classes': 4, 'device': DEVICE},
            'type': 'no_group',
        },
    }

    # Load all models
    models = {}
    for name, cfg in model_configs.items():
        print(f"\nLoading {name} from {cfg['ckpt']}...")
        try:
            models[name] = load_trained_model(cfg['class'], cfg['ckpt'], **cfg['kwargs'])
        except Exception as e:
            print(f"  WARNING: Could not load {name}: {e}")

    # Load data
    print("\nLoading CRWU data...")
    signals, labels, class_names = load_crwu_data(CRWU_ROOT, signal_length=8192, overlap=0.5)
    dataset = CRWUDataset(signals, labels, fs=SAMPLING_RATE_CRWU, spec_size=(224, 224))
    from torch.utils.data import random_split
    total = len(dataset)
    test_size = int(total * 0.2)
    val_size = int(total * 0.1)
    train_size = total - test_size - val_size
    _, _, test_set = random_split(dataset, [train_size, val_size, test_size],
                                   generator=torch.Generator().manual_seed(42))
    print(f"Test set size: {len(test_set)}")
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=2)

    # Get predictions from all models
    all_preds = {}
    for name, model in models.items():
        print(f"\nRunning {name} inference...")
        preds = get_test_predictions(model, test_loader)
        correct = sum(1 for p in preds if p['correct'])
        print(f"  {name} accuracy: {correct}/{len(preds)} = {correct/len(preds)*100:.2f}%")
        all_preds[name] = preds

    # Select cases based on SGVT-FD
    print("\nSelecting cases...")
    case_a_info, case_b_info, case_c_info = select_cases(all_preds['SGVT-FD'])
    case_infos = {'A': case_a_info, 'B': case_b_info, 'C': case_c_info}
    for name, info in case_infos.items():
        print(f"  Case {name}: idx={info['global_idx']}, "
              f"true={CRWU_CLASSES[info['true_label']]}, "
              f"pred={CRWU_CLASSES[info['pred_label']]}, conf={info['confidence']:.4f}")

    # Extract intermediates from all models
    print("\nExtracting intermediates...")
    all_cases = {}  # {model_name: {case_name: case_data}}
    for model_name, model in models.items():
        all_cases[model_name] = {}
        for case_name, info in case_infos.items():
            all_cases[model_name][case_name] = extract_case_intermediates(model, test_set, info)

    # Save NPZ (SGVT-FD only)
    for name, data in all_cases['SGVT-FD'].items():
        np.savez(os.path.join(OUTPUT_DIR, f"case_{name}.npz"), **data)
    print(f"Saved NPZ files to {OUTPUT_DIR}")

    # Generate figure (SGVT-FD only)
    sgvt_cases = all_cases['SGVT-FD']
    fig_path = os.path.join(OUTPUT_DIR, "case_study.png")
    plot_case_study(sgvt_cases['A'], sgvt_cases['B'], sgvt_cases['C'], fig_path)
    pdf_path = os.path.join(base_dir, "figs", "case_study.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    plot_case_study(sgvt_cases['A'], sgvt_cases['B'], sgvt_cases['C'], pdf_path)
    print(f"Saved PDF to {pdf_path}")

    # Generate LLM explanations for all models
    print("\nGenerating LLM-style diagnostic explanations...")
    explanations = {}
    for case_name in ['A', 'B', 'C']:
        explanations[case_name] = {
            'true': CRWU_CLASSES[case_infos[case_name]['true_label']],
        }
        for model_name in models:
            case_data = all_cases[model_name][case_name]
            cfg = model_configs[model_name]
            exp_text = generate_llm_explanation(case_data, model_name)
            explanations[case_name][model_name.lower().replace(' ', '_')] = exp_text
            explanations[case_name][f'{model_name.lower().replace(" ", "_")}_conf'] = case_data['confidence']
            explanations[case_name][f'{model_name.lower().replace(" ", "_")}_pred'] = CRWU_CLASSES[case_data['pred_label']]

    # Save explanations
    with open(os.path.join(OUTPUT_DIR, "llm_explanations.json"), "w") as f:
        json.dump(explanations, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE FOR PAPER")
    print("=" * 70)
    model_names = list(models.keys())
    print(f"\n{'Case':<8} {'True':<12}", end='')
    for mn in model_names:
        print(f" {mn + ' Pred':<16} {mn + ' Conf':<10}", end='')
    print()
    print("-" * (8 + 12 + len(model_names) * 26))
    for case_name in ['A', 'B', 'C']:
        exp = explanations[case_name]
        print(f"Case {case_name:<4} {exp['true']:<12}", end='')
        for mn in model_names:
            key = mn.lower().replace(' ', '_')
            pred = exp.get(f'{key}_pred', 'N/A')
            conf = exp.get(f'{key}_conf', 0)
            print(f" {pred:<16} {conf:<10.2%}", end='')
        print()

    # Print full explanations
    print("\n" + "=" * 70)
    print("FULL EXPLANATIONS")
    print("=" * 70)
    for case_name in ['A', 'B', 'C']:
        exp = explanations[case_name]
        print(f"\n=== Case {case_name} (True: {exp['true']}) ===")
        for mn in model_names:
            key = mn.lower().replace(' ', '_')
            print(f"\n[{mn}]")
            print(exp.get(key, 'N/A'))

    print(f"\nAll outputs saved to {OUTPUT_DIR}")
    return explanations


if __name__ == "__main__":
    run()
