"""DREAM-RNN — the Prix Fixe composite from Rafi et al. 2024 (Nat Biotechnol;
Random Promoter DREAM Challenge 2022).

This is a **single** network, not an ensemble. The DREAM authors searched all
combinations of the top-three teams' interchangeable blocks; DREAM-RNN is the
best combination built around the BHI team's recurrent core:

    first  — BHI multi-kernel conv      (kernels 9, 15; 320 ch)
    core   — BHI bidirectional LSTM      (hidden 320/dir → 640) + multi-kernel conv
    final  — Autosome 18-bin soft-classification head → expected-value scalar

The block class names here are deliberately neutral (no team prefixes) — see
``benchmarks/rafi_mpra_promoter.md``. The **module/attribute hierarchy**,
however, is preserved exactly (``first`` / ``core`` / ``final``, with
``conv_list`` / ``lstm`` / ``do`` / ``mapper`` / ``bins`` beneath) so the
published ``model_best.pth`` state_dict (Zenodo 10633252, dir ``0_1_1_0``)
loads with ``strict=True`` unchanged. (``0_1_1_0`` = DREAM-RNN, verified by a
strict load: it carries ``core.lstm.*`` and final in_channels 320; ``0_1_0_0``
is DREAM-CNN — an Autosome conv core — despite the notebook comments.)

Ported from the de-Boer-Lab reference implementation (MIT). The DREAM-RNN
assembly (block types + hyperparameters) matches
``2_predict_seq_pos.py``'s ``DREAM_RNN`` branch and the
``DREAMNets_BuildModel_Train_and_Predict.ipynb`` notebook. seqsize does not
affect any parameter shape in these three blocks, so it is not a constructor
argument here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Conv1d(padding='same') → ReLU → MaxPool → Dropout. (BHI ``ConvBlock``.)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding="same"
        )
        self.mp = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.do(self.mp(F.relu(self.conv(x))))


class _FirstBlock(nn.Module):
    """BHI first-layer block: parallel multi-kernel conv, concatenated on channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        pool_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert out_channels % len(kernel_sizes) == 0
        each = out_channels // len(kernel_sizes)
        self.conv_list = nn.ModuleList(
            [_ConvBlock(in_channels, each, k, pool_size, dropout) for k in kernel_sizes]
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([conv(x) for conv in self.conv_list], dim=1)


class _CoreBlock(nn.Module):
    """BHI core block: bidirectional LSTM → parallel multi-kernel conv → dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lstm_hidden_channels: int,
        kernel_sizes: list[int],
        pool_size: int,
        dropout1: float,
        dropout2: float,
    ) -> None:
        super().__init__()
        assert out_channels % len(kernel_sizes) == 0
        each = out_channels // len(kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden_channels,
            batch_first=True,
            bidirectional=True,
        )
        self.conv_list = nn.ModuleList(
            [
                _ConvBlock(2 * lstm_hidden_channels, each, k, pool_size, dropout1)
                for k in kernel_sizes
            ]
        )
        self.do = nn.Dropout(dropout2)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, L, C)
        x, _ = self.lstm(x)  # (B, L, 2*hidden)
        x = x.permute(0, 2, 1)  # (B, 2*hidden, L)
        x = torch.cat([conv(x) for conv in self.conv_list], dim=1)
        return self.do(x)


class _FinalBlock(nn.Module):
    """Autosome final block: 1×1 conv → N_BINS → global avg-pool → softmax →
    expected bin value (a single scalar per sequence)."""

    N_BINS = 18

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # nn.Sequential with a single Conv1d (matches upstream ``mapper.0.*`` keys;
        # the upstream SiLU is commented out, so it is omitted here too).
        self.mapper = nn.Sequential(
            nn.Conv1d(in_channels, self.N_BINS, kernel_size=1, padding="same"),
        )
        self.register_buffer(
            "bins",
            torch.arange(start=0, end=self.N_BINS, step=1, requires_grad=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)  # (B, N_BINS)
        x = F.softmax(x, dim=1)
        return (x * self.bins).sum(dim=1)  # (B,)


class DreamRnn(nn.Module):
    """DREAM-RNN. Defaults reproduce the published ``0_1_1_0`` checkpoint.

    Input: ``(B, 6, L)`` — 4 one-hot base channels + is_reverse + is_singleton.
    Output: ``(B,)`` — predicted reporter expression (expected value over 18 bins).
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 320,
        lstm_hidden_channels: int = 320,
        kernel_sizes: tuple[int, ...] = (9, 15),
    ) -> None:
        super().__init__()
        ks = list(kernel_sizes)
        self.first = _FirstBlock(
            in_channels, out_channels, ks, pool_size=1, dropout=0.2
        )
        self.core = _CoreBlock(
            self.first.out_channels,
            out_channels,
            lstm_hidden_channels,
            ks,
            pool_size=1,
            dropout1=0.2,
            dropout2=0.5,
        )
        self.final = _FinalBlock(self.core.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        x = self.core(x)
        return self.final(x)
