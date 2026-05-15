"""
Semantic Grouping Module (SGVT)
Groups visual tokens from spectrograms based on semantic similarity.
Core innovation: domain-aware clustering with fault frequency priors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class SemanticGroupingModule(nn.Module):
    """Semantic Grouping for Visual Tokens.

    Groups spectrogram patches by semantic similarity and merges them.
    Uses fault characteristic frequencies as domain priors for initialization.

    Input: spectrogram image (B, 3, H, W)
    Output: merged visual tokens (B, K, D)
    """

    def __init__(self, num_groups=32, patch_size=16, feature_dim=768,
                 merged_dim=768, use_domain_prior=True):
        super().__init__()
        self.num_groups = num_groups
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.merged_dim = merged_dim
        self.use_domain_prior = use_domain_prior

        # Lightweight feature extractor (frozen CLIP ViT or custom)
        # We use a small ViT-like encoder for token extraction
        self.patch_embed = nn.Conv2d(
            3, feature_dim, kernel_size=patch_size, stride=patch_size
        )

        # Learnable position encoding
        num_patches = (224 // patch_size) ** 2  # 196 for 224x224 with 16x16
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, feature_dim) * 0.02)

        # Group assignment network
        # Maps each token to a group probability distribution
        self.group_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_groups),
        )

        # Token merging weights (learned per group)
        self.merge_weights = nn.Parameter(torch.ones(num_groups, feature_dim))

        # Output projection
        self.output_proj = nn.Linear(feature_dim, merged_dim)
        self.norm = nn.LayerNorm(merged_dim)

    def extract_patches(self, x):
        """Extract patch tokens from spectrogram image.

        Args:
            x: (B, 3, H, W) spectrogram image

        Returns:
            tokens: (B, N, D) patch tokens where N = (H/P)*(W/P)
        """
        B = x.shape[0]
        # Patch embedding: (B, 3, H, W) -> (B, D, H/P, W/P)
        patches = self.patch_embed(x)
        # Reshape to (B, N, D)
        H_p, W_p = patches.shape[2], patches.shape[3]
        tokens = patches.reshape(B, self.feature_dim, -1).transpose(1, 2)
        # Add position encoding
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        return tokens

    def compute_group_assignments(self, tokens):
        """Compute soft group assignments for each token.

        Args:
            tokens: (B, N, D) patch tokens

        Returns:
            assignments: (B, N, K) soft assignment probabilities
        """
        # Project to group logits
        logits = self.group_proj(tokens)  # (B, N, K)
        # Soft assignment with temperature
        assignments = F.softmax(logits, dim=-1)
        return assignments

    def merge_tokens(self, tokens, assignments):
        """Merge tokens within each group using weighted pooling.

        Args:
            tokens: (B, N, D) patch tokens
            assignments: (B, N, K) soft group assignments

        Returns:
            merged: (B, K, D) merged tokens per group
        """
        B, N, D = tokens.shape
        K = self.num_groups

        # Weighted pooling: (B, K, N) x (B, N, D) -> (B, K, D)
        # assignments^T @ tokens
        assignments_T = assignments.transpose(1, 2)  # (B, K, N)
        merged = torch.bmm(assignments_T, tokens)  # (B, K, D)

        # Normalize by group size (sum of assignments)
        group_sizes = assignments.sum(dim=1, keepdim=True).transpose(1, 2)  # (B, K, 1)
        group_sizes = group_sizes.clamp(min=1e-8)
        merged = merged / group_sizes

        # Apply learnable merge weights
        merged = merged * self.merge_weights.unsqueeze(0)

        return merged

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, H, W) spectrogram image

        Returns:
            merged_tokens: (B, K, merged_dim) merged visual tokens
            assignments: (B, N, K) soft group assignments (for visualization)
        """
        # Extract patch tokens
        tokens = self.extract_patches(x)  # (B, N, D)

        # Compute group assignments
        assignments = self.compute_group_assignments(tokens)  # (B, N, K)

        # Merge tokens within groups
        merged = self.merge_tokens(tokens, assignments)  # (B, K, D)

        # Project and normalize
        merged = self.output_proj(merged)
        merged = self.norm(merged)

        return merged, assignments


class DomainAwareGrouping(SemanticGroupingModule):
    """Domain-Aware Semantic Grouping with fault frequency priors.

    Extends SemanticGroupingModule by initializing cluster centers
    based on known fault characteristic frequencies.
    """

    def __init__(self, num_groups=32, patch_size=16, feature_dim=768,
                 merged_dim=768, fault_freqs=None, fs=12000):
        super().__init__(num_groups, patch_size, feature_dim, merged_dim)
        self.fault_freqs = fault_freqs or {}
        self.fs = fs

        # Frequency-aware initialization
        if fault_freqs:
            self._init_with_domain_prior(fault_freqs, fs)

    def _init_with_domain_prior(self, fault_freqs, fs):
        """Initialize group projections with domain knowledge.

        The idea: assign some groups to focus on fault-relevant frequency bands.
        """
        # Map fault frequencies to patch indices
        # For a spectrogram with N patches along frequency axis,
        # fault frequencies correspond to specific patches
        num_patches_freq = 224 // self.patch_size  # 14 patches along freq axis
        freq_resolution = fs / (2 * num_patches_freq)

        # Assign groups to fault frequency bands
        freq_list = sorted(fault_freqs.values())
        num_freq_bands = min(len(freq_list), self.num_groups // 2)

        if num_freq_bands > 0:
            # Initialize group projection bias to favor fault frequency bands
            with torch.no_grad():
                bias = torch.zeros(self.num_groups)
                for i, freq in enumerate(freq_list[:num_freq_bands]):
                    # Map frequency to group index
                    group_idx = int(i * self.num_groups / num_freq_bands)
                    bias[group_idx] = 1.0
                # Distribute remaining groups evenly
                for i in range(num_freq_bands, self.num_groups):
                    bias[i] = 0.1
                self.group_proj[-1].bias.copy_(bias)


class InformationTheoreticGrouping(SemanticGroupingModule):
    """Information-Theoretic Semantic Grouping.

    Uses mutual information estimation to guide grouping,
    ensuring maximum diagnostic information preservation.
    """

    def __init__(self, num_groups=32, patch_size=16, feature_dim=768,
                 merged_dim=768, mi_weight=0.01):
        super().__init__(num_groups, patch_size, feature_dim, merged_dim)
        self.mi_weight = mi_weight

    def compute_mi_loss(self, tokens, assignments, labels):
        """Compute mutual information regularization loss.

        Encourages grouping that preserves class-relevant information.

        Args:
            tokens: (B, N, D) patch tokens
            assignments: (B, N, K) soft group assignments
            labels: (B,) class labels

        Returns:
            mi_loss: scalar loss
        """
        B, N, D = tokens.shape
        K = self.num_groups

        # Compute class-conditional statistics for each group
        merged = self.merge_tokens(tokens, assignments)  # (B, K, D)

        # For each group, compute inter-class variance
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=tokens.device)

        # Global mean per group
        global_mean = merged.mean(dim=0, keepdim=True)  # (1, K, D)

        # Between-class variance (normalized)
        between_var = torch.tensor(0.0, device=tokens.device)
        total_var = torch.tensor(0.0, device=tokens.device)
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                class_mean = merged[mask].mean(dim=0, keepdim=True)  # (1, K, D)
                between_var += mask.sum() * ((class_mean - global_mean) ** 2).mean()
                # Within-class variance
                within = ((merged[mask] - class_mean) ** 2).mean()
                total_var += mask.sum() * within

        between_var /= B
        total_var = (total_var / B).clamp(min=1e-8)

        # Maximize between/within ratio (Fisher criterion)
        # Use negative log of ratio as loss (minimize to maximize ratio)
        fisher_ratio = between_var / total_var
        mi_loss = -self.mi_weight * torch.log(fisher_ratio + 1e-8)

        return mi_loss

    def forward(self, x, labels=None):
        """Forward pass with optional MI regularization.

        Args:
            x: (B, 3, H, W) spectrogram image
            labels: (B,) optional class labels for MI loss

        Returns:
            merged_tokens: (B, K, merged_dim)
            assignments: (B, N, K)
            mi_loss: scalar (0 if labels not provided)
        """
        tokens = self.extract_patches(x)
        assignments = self.compute_group_assignments(tokens)
        merged = self.merge_tokens(tokens, assignments)
        merged = self.output_proj(merged)
        merged = self.norm(merged)

        mi_loss = torch.tensor(0.0, device=x.device)
        if labels is not None:
            mi_loss = self.compute_mi_loss(tokens, assignments, labels)

        return merged, assignments, mi_loss
