"""
SGVT-FD: Semantic Grouped Visual Tokens for Fault Diagnosis
Main model combining semantic grouping with VLM classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .semantic_grouping import SemanticGroupingModule, DomainAwareGrouping, InformationTheoreticGrouping
from .vlm_inference import SimpleVLMClassifier


class SGVTModel(nn.Module):
    """SGVT-FD: Semantic Grouped Visual Tokens for Fault Diagnosis.

    Pipeline:
    1. Spectrogram image -> patch tokens
    2. Semantic grouping: cluster tokens by similarity
    3. Merge tokens within groups
    4. VLM classification (CLIP + MLP head for training, Qwen for inference)

    Args:
        num_classes: Number of fault classes
        num_groups: Number of semantic groups (K)
        patch_size: ViT patch size
        feature_dim: Token feature dimension
        use_domain_prior: Whether to use fault frequency priors
        use_mi_loss: Whether to use information-theoretic loss
        fault_freqs: Dictionary of fault characteristic frequencies
        fs: Sampling frequency
        device: Device to use
    """

    def __init__(self, num_classes=4, num_groups=32, patch_size=16,
                 feature_dim=768, use_domain_prior=True, use_mi_loss=True,
                 fault_freqs=None, fs=12000, device="cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.device = device

        # Semantic grouping module
        if use_mi_loss:
            self.grouping = InformationTheoreticGrouping(
                num_groups=num_groups,
                patch_size=patch_size,
                feature_dim=feature_dim,
                merged_dim=feature_dim,
            )
        elif use_domain_prior and fault_freqs:
            self.grouping = DomainAwareGrouping(
                num_groups=num_groups,
                patch_size=patch_size,
                feature_dim=feature_dim,
                merged_dim=feature_dim,
                fault_freqs=fault_freqs,
                fs=fs,
            )
        else:
            self.grouping = SemanticGroupingModule(
                num_groups=num_groups,
                patch_size=patch_size,
                feature_dim=feature_dim,
                merged_dim=feature_dim,
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        # Group diversity loss weight
        self.diversity_weight = 0.01

    def compute_diversity_loss(self, assignments):
        """Encourage diverse group assignments (avoid collapse).

        Args:
            assignments: (B, N, K) soft group assignments

        Returns:
            diversity_loss: scalar
        """
        # Average assignment per group
        avg_assign = assignments.mean(dim=(0, 1))  # (K,)
        # Entropy of group distribution (higher = more diverse)
        entropy = -(avg_assign * torch.log(avg_assign + 1e-8)).sum()
        # Maximize entropy (negative because we minimize loss)
        return -entropy

    def forward(self, x, labels=None):
        """Forward pass.

        Args:
            x: (B, 3, H, W) spectrogram images
            labels: (B,) optional class labels

        Returns:
            logits: (B, num_classes)
            loss: scalar (if labels provided)
            assignments: (B, N, K) group assignments (for visualization)
        """
        # Semantic grouping
        if isinstance(self.grouping, InformationTheoreticGrouping):
            merged_tokens, assignments, mi_loss = self.grouping(x, labels)
        else:
            merged_tokens, assignments = self.grouping(x)
            mi_loss = torch.tensor(0.0, device=x.device)

        # Mean pooling over groups
        pooled = merged_tokens.mean(dim=1)  # (B, feature_dim)

        # Classification
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            div_loss = self.compute_diversity_loss(assignments)
            loss = ce_loss + mi_loss + self.diversity_weight * div_loss

        return logits, loss, assignments

    def predict(self, x):
        """Predict class for input spectrograms.

        Args:
            x: (B, 3, H, W) spectrogram images

        Returns:
            preds: (B,) predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(x)
            preds = logits.argmax(dim=1)
        return preds


class SGVTBaseline(nn.Module):
    """Baseline: same architecture but with random/fixed grouping (no semantic).

    Used for ablation studies.
    """

    def __init__(self, num_classes=4, num_groups=32, patch_size=16,
                 feature_dim=768, grouping_type="random", device="cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.grouping_type = grouping_type
        self.device = device

        # Same patch embedding as SGVT
        self.patch_embed = nn.Conv2d(
            3, feature_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, feature_dim) * 0.02)

        # Fixed grouping (no learning)
        self.feature_dim = feature_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def _fixed_grouping(self, tokens):
        """Apply fixed (random or uniform) grouping.

        Args:
            tokens: (B, N, D)

        Returns:
            merged: (B, K, D)
        """
        B, N, D = tokens.shape
        K = self.num_groups

        if self.grouping_type == "random":
            # Random assignment (fixed across batches)
            if not hasattr(self, '_random_assignments'):
                assignments = torch.zeros(N, K)
                for i in range(N):
                    assignments[i, i % K] = 1.0
                self._random_assignments = assignments.to(self.device)
            assign = self._random_assignments.unsqueeze(0).expand(B, -1, -1)

        elif self.grouping_type == "uniform":
            # Uniform pooling (all tokens in one group)
            assign = torch.ones(B, N, K, device=self.device) / K

        else:
            # Sequential grouping
            assign = torch.zeros(B, N, K, device=self.device)
            group_size = N // K
            for k in range(K):
                start = k * group_size
                end = min(start + group_size, N)
                assign[:, start:end, k] = 1.0

        # Weighted pooling
        assign_T = assign.transpose(1, 2)
        merged = torch.bmm(assign_T, tokens)
        group_sizes = assign.sum(dim=1, keepdim=True).transpose(1, 2).clamp(min=1e-8)
        merged = merged / group_sizes

        return merged

    def forward(self, x, labels=None):
        """Forward pass with fixed grouping."""
        B = x.shape[0]
        patches = self.patch_embed(x)
        tokens = patches.reshape(B, self.feature_dim, -1).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        merged = self._fixed_grouping(tokens)
        pooled = merged.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

        return logits, loss, None


class NoGroupingBaseline(nn.Module):
    """Baseline: no grouping, use all tokens (standard ViT approach)."""

    def __init__(self, num_classes=4, patch_size=16, feature_dim=768, device="cuda"):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim

        self.patch_embed = nn.Conv2d(
            3, feature_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, feature_dim) * 0.02)

        # Transformer encoder for processing all tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=8, dim_feedforward=feature_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, labels=None):
        B = x.shape[0]
        patches = self.patch_embed(x)
        tokens = patches.reshape(B, self.feature_dim, -1).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Transformer encoding
        encoded = self.transformer(tokens)

        # CLS token output
        cls_output = encoded[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

        return logits, loss, None
