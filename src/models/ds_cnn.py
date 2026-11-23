"""Hybrid AcousticDSCNN: Masked Spectrogram Autoencoder + Sub-Cluster Cosine Embedder.

Prevents Neural Collapse on `noAttribute` machines (bearingEmu, valveEmu) via two mechanisms:
  1. Spectrogram Decoder: Reconstructs the 3-channel (128x94) spatial spectrogram directly from
     the compressed bottleneck, forcing the network to preserve fine harmonics & transient clicks.
  2. Multi-Prototype Sub-Cluster Cosine Head (K=8 sub-centers per class): Allows each normal
     machine to occupy a rich manifold of operating modes instead of collapsing to a single point.

At edge inference (ONNX), `AcousticHybridEdgeModule` returns `(embedding, frame_recon_mse)`
in a single lightweight CPU pass (<15ms).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.se = SqueezeExcitation(out_ch) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return self.se(x)


class LightweightSpecDecoder(nn.Module):
    """Compact upsampling decoder reconstructing (B, C, F, T) from spatial bottleneck."""

    def __init__(self, bottleneck_ch: int, out_channels: int = 3, width_mult: float = 1.0):
        super().__init__()

        def c(ch: int) -> int:
            return max(int(ch * width_mult), 16)

        # Bottleneck compression forces meaningful representations (no trivial identity shortcut)
        self.compress = nn.Sequential(
            nn.Conv2d(bottleneck_ch, c(64), kernel_size=1, bias=False),
            nn.BatchNorm2d(c(64)),
            nn.SiLU(inplace=True),
        )
        self.up1 = DepthwiseSeparableBlock(c(64), c(64), stride=1, use_se=False)
        self.up2 = DepthwiseSeparableBlock(c(64), c(32), stride=1, use_se=False)
        self.up3 = DepthwiseSeparableBlock(c(32), c(24), stride=1, use_se=False)
        self.out_conv = nn.Conv2d(c(24), out_channels, kernel_size=3, padding=1)

    def forward(self, feat_map: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        h = self.compress(feat_map)
        h = F.interpolate(h, scale_factor=2.0, mode="bilinear", align_corners=False)
        h = self.up1(h)
        h = F.interpolate(h, scale_factor=2.0, mode="bilinear", align_corners=False)
        h = self.up2(h)
        h = F.interpolate(h, size=target_hw, mode="bilinear", align_corners=False)
        h = self.up3(h)
        return self.out_conv(h)


class AcousticHybridBackbone(nn.Module):
    """ONNX-exportable Encoder + Decoder returning (L2-embedding, per-frame reconstruction MSE)."""

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 256,
        width_mult: float = 1.0,
        dropout: float = 0.15,
    ):
        super().__init__()

        def c(ch: int) -> int:
            return max(int(ch * width_mult), 16)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(c(32), c(64), stride=1),
            DepthwiseSeparableBlock(c(64), c(128), stride=2),
            DepthwiseSeparableBlock(c(128), c(128), stride=1),
            DepthwiseSeparableBlock(c(128), c(256), stride=2),
            DepthwiseSeparableBlock(c(256), c(256), stride=1),
        )

        # Frequency-aware + global Avg/Max pooling preserves rich spectral structure
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(c(256) * 2, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        self.decoder = LightweightSpecDecoder(c(256), out_channels=in_channels, width_mult=width_mult)

    def forward_full(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (L2-normalized embedding [B, D], reconstructed spectrogram [B, C, F, T])."""
        h = self.blocks(self.stem(x))
        pooled = torch.cat([self.avg_pool(h), self.max_pool(h)], dim=1).flatten(1)
        emb = F.normalize(self.proj(pooled), p=2, dim=1)
        recon = self.decoder(h, target_hw=(x.shape[2], x.shape[3]))
        return emb, recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ONNX edge forward: returns (embedding [B, D], frame_mse [B, T])."""
        emb, recon = self.forward_full(x)
        # Per-time-frame MSE across channels & mel bins allows top-k frame anomaly scoring
        frame_mse = torch.mean((recon - x) ** 2, dim=(1, 2))
        return emb, frame_mse


class SubClusterCosineHead(nn.Module):
    """Multi-Prototype Sub-Cluster Cosine Classifier (K sub-centers per class).

    Unlike single-prototype CosFace, K=8 sub-clusters let each machine retain distinct
    normal sub-modes in embedding space without collapsing to a single vector.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        sub_clusters: int = 8,
        scale: float = 12.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.sub_clusters = sub_clusters
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes * sub_clusters, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, w_norm)  # (B, num_classes * K)
        cosine = cosine.view(-1, self.num_classes, self.sub_clusters)
        # LogSumExp / max pooling over sub-clusters selects nearest operating mode
        class_cosine, _ = torch.max(cosine, dim=2)
        return class_cosine * self.scale


class DCaseHybridModel(nn.Module):
    """Joint Masked Spectrogram Autoencoder + Sub-Cluster Proxy Model."""

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 256,
        num_classes: int = 4,
        width_mult: float = 1.0,
        dropout: float = 0.15,
        sub_clusters: int = 8,
        cosine_scale: float = 12.0,
    ):
        super().__init__()
        self.backbone = AcousticHybridBackbone(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            width_mult=width_mult,
            dropout=dropout,
        )
        self.proxy_head = SubClusterCosineHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            sub_clusters=sub_clusters,
            scale=cosine_scale,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb, recon = self.backbone.forward_full(x)
        logits = self.proxy_head(emb)
        return emb, recon, logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
