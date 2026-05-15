"""
Baseline models for fault diagnosis comparison.
CNN, LSTM, ResNet, ViT baselines.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    """1D CNN baseline for vibration signal classification."""

    def __init__(self, num_classes=4, signal_length=8192):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, labels=None):
        # x: (B, 3, H, W) spectrogram - convert to 1D
        if x.dim() == 4:
            # Take mean across channels and flatten
            x = x.mean(dim=1)  # (B, H, W)
            x = x.mean(dim=1, keepdim=True)  # (B, 1, W) - simplified

        features = self.features(x)
        features = features.squeeze(-1)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss, None


class LSTMBaseline(nn.Module):
    """LSTM baseline for vibration signal classification."""

    def __init__(self, num_classes=4, input_dim=224, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, labels=None):
        if x.dim() == 4:
            # (B, 3, H, W) -> (B, W, H) treat as sequence
            x = x.mean(dim=1)  # (B, H, W)
            x = x.permute(0, 2, 1)  # (B, W, H)

        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        hidden = lstm_out[:, -1, :]
        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss, None


class ResNetBaseline(nn.Module):
    """ResNet-18 baseline for spectrogram classification."""

    def __init__(self, num_classes=4):
        super().__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x, labels=None):
        logits = self.backbone(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss, None


class ViTBaseline(nn.Module):
    """Vision Transformer baseline for spectrogram classification."""

    def __init__(self, num_classes=4, patch_size=16, feature_dim=768,
                 num_heads=8, num_layers=6, image_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            3, feature_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, feature_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads,
            dim_feedforward=feature_dim * 4, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, labels=None):
        B = x.shape[0]
        patches = self.patch_embed(x)  # (B, D, H/P, W/P)
        tokens = patches.reshape(B, self.feature_dim, -1).transpose(1, 2)  # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        encoded = self.transformer(tokens)
        encoded = self.norm(encoded)

        cls_output = encoded[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        return logits, loss, None


class ToMeBaseline(nn.Module):
    """ToMe-like (Token Merging, ICLR 2023) token compression baseline.

    Bipartite soft matching: repeatedly merges the most similar pair of tokens
    until reaching target token count.
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
            # Split tokens into two halves (A and B)
            half = N_cur // 2
            tokens_a = tokens[:, :half, :]  # (B, half, D)
            tokens_b = tokens[:, half:, :]  # (B, N_cur-half, D)

            # Compute cosine similarity between A and B
            a_norm = F.normalize(tokens_a, dim=-1)
            b_norm = F.normalize(tokens_b, dim=-1)
            sim = torch.bmm(a_norm, b_norm.transpose(1, 2))  # (B, half, N_cur-half)

            # For each token in A, find best match in B
            _, best_b_idx = sim.max(dim=2)  # (B, half)

            # Merge matched pairs
            merged = (tokens_a + tokens_b.gather(1,
                best_b_idx.unsqueeze(-1).expand(-1, -1, D))) / 2.0  # (B, half, D)

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
    """TokenLearner-like learnable token selection baseline.

    Learns K weighted combinations of input tokens via a lightweight MLP,
    producing a compact set of informative tokens.
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
        weights = self.token_weight(tokens)
        weights = F.softmax(weights, dim=1)
        merged = torch.bmm(weights.transpose(1, 2), tokens)
        pooled = merged.mean(dim=1)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        return logits, loss, None


class CVTBaseline(nn.Module):
    """CVT baseline: Correlation-based Visual Tokenization.

    Groups tokens by pairwise correlation instead of semantic similarity.
    Serves as a baseline version of our SGVT-FD framework,
    replacing learned semantic grouping with simple correlation-based grouping.
    """

    def __init__(self, num_classes=4, num_groups=32, patch_size=16,
                 feature_dim=768, device="cuda"):
        super().__init__()
        self.num_groups = num_groups
        self.device = device
        self.feature_dim = feature_dim

        self.patch_embed = nn.Conv2d(
            3, feature_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, feature_dim) * 0.02)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def _correlation_grouping(self, tokens):
        """Group tokens by pairwise correlation (CVT-style).

        Args:
            tokens: (B, N, D)

        Returns:
            merged: (B, K, D)
        """
        B, N, D = tokens.shape
        K = self.num_groups

        # Compute pairwise correlation
        tokens_norm = F.normalize(tokens, dim=-1)
        correlation = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # (B, N, N)

        # Correlation-based grouping
        # Select K representative tokens (highest average correlation)
        avg_corr = correlation.mean(dim=2)  # (B, N)
        _, topk_indices = avg_corr.topk(K, dim=1)  # (B, K)

        # Assign each token to most correlated representative
        rep_corr = torch.gather(
            correlation, 1,
            topk_indices.unsqueeze(2).expand(-1, -1, N)
        )  # (B, K, N)
        assignments = rep_corr.transpose(1, 2)  # (B, N, K)
        assignments = F.softmax(assignments * 10, dim=-1)  # sharpen

        # Weighted pooling
        assign_T = assignments.transpose(1, 2)
        merged = torch.bmm(assign_T, tokens)
        group_sizes = assignments.sum(dim=1, keepdim=True).transpose(1, 2).clamp(min=1e-8)
        merged = merged / group_sizes

        return merged

    def forward(self, x, labels=None):
        B = x.shape[0]
        patches = self.patch_embed(x)
        tokens = patches.reshape(B, self.feature_dim, -1).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        merged = self._correlation_grouping(tokens)
        pooled = merged.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        return logits, loss, None
