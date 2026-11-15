"""Stage 11: Temporal Deep Learning Architectures for Dynamic J/Z Gesture Recognition.

Provides two lightweight (< 50k params) sequence classification models:
1. TemporalConvNet (1D-TCN): Dilated residual 1D temporal convolutions.
2. BiGRUNet (Bi-GRU): Bidirectional Gated Recurrent Unit with temporal pooling.

Input specification:
    Shape: (batch_size, sequence_length=16, feature_dim=69)
Output specification:
    Logits: (batch_size, num_classes=2) -> index 0: 'J', index 1: 'Z'
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ["J", "Z"]
LABEL_TO_IDX = {"J": 0, "Z": 1}
IDX_TO_LABEL = {0: "J", 1: "Z"}


# ==============================================================================
# MODEL A: 1D-TCN (Temporal Convolutional Network)
# ==============================================================================

class TemporalResidualBlock(nn.Module):
    """Residual block with dilated 1D temporal convolutions and batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else None
        )
        self.bn_res = nn.BatchNorm1d(out_channels) if in_channels != out_channels else None
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        if self.downsample is not None and self.bn_res is not None:
            res = self.bn_res(self.downsample(x))

        out = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.bn2(self.conv2(out))))
        return self.relu_out(out + res)


class TemporalConvNet(nn.Module):
    """1D Temporal Convolutional Network (< 50,000 parameters) for J/Z sequence classification."""

    def __init__(
        self,
        input_dim: int = 69,
        num_classes: int = 2,
        hidden_channels: int = 48,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.in_proj = nn.Conv1d(input_dim, hidden_channels, 1, bias=False)
        self.in_bn = nn.BatchNorm1d(hidden_channels)
        self.in_relu = nn.ReLU(inplace=True)

        self.block1 = TemporalResidualBlock(
            hidden_channels, hidden_channels, kernel_size=3, dilation=1, dropout=dropout
        )
        self.block2 = TemporalResidualBlock(
            hidden_channels, hidden_channels, kernel_size=3, dilation=2, dropout=dropout
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input: (B, 16, 69) -> transposed to (B, 69, 16) for Conv1d.
        Output: (B, 2) logits.
        """
        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.in_relu(self.in_bn(self.in_proj(x)))
        x = self.block1(x)
        x = self.block2(x)
        feat = self.pool(x).squeeze(-1)
        return self.fc(feat)


# ==============================================================================
# MODEL B: Bi-GRU (Bidirectional Gated Recurrent Unit)
# ==============================================================================

class BiGRUNet(nn.Module):
    """Bidirectional GRU (< 50,000 parameters) for J/Z sequence classification."""

    def __init__(
        self,
        input_dim: int = 69,
        hidden_size: int = 48,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input: (B, 16, 69)
        Output: (B, 2) logits.
        """
        out, _ = self.gru(x)  # (B, 16, hidden*2)
        # Temporal average pooling
        feat = torch.mean(out, dim=1)
        feat = self.dropout(feat)
        return self.fc(feat)


# ==============================================================================
# INFERENCE WRAPPER FOR BENCHMARKING
# ==============================================================================

@dataclass(frozen=True)
class DeepLearningInferenceResult:
    """Output contract for deep learning inference."""
    prediction: str                       # 'J' or 'Z'
    confidence: float                     # 0.0 .. 1.0
    prob_j: float                         # P(J)
    prob_z: float                         # P(Z)
    margin: float                         # |prob_j - prob_z|
    latency_ms: float = 0.0


class DeepLearningPredictor:
    """Inference runner for trained PyTorch checkpoints."""

    def __init__(self, model: nn.Module, checkpoint_path: str | None = None) -> None:
        self.model = model
        self.model.eval()
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state)
            self.model.eval()

    def predict(self, sequence: Sequence[Sequence[float]] | np.ndarray) -> DeepLearningInferenceResult:
        t0 = time.perf_counter()
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        tensor = torch.from_numpy(arr)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

        prob_j = float(probs[0])
        prob_z = float(probs[1])
        pred_idx = int(np.argmax(probs))
        pred = IDX_TO_LABEL[pred_idx]
        conf = float(probs[pred_idx])
        margin = float(abs(prob_j - prob_z))
        lat = (time.perf_counter() - t0) * 1000.0

        return DeepLearningInferenceResult(
            prediction=pred,
            confidence=conf,
            prob_j=prob_j,
            prob_z=prob_z,
            margin=margin,
            latency_ms=lat,
        )


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
