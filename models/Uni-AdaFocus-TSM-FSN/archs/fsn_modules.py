"""FSN-specific, opt-in modules for Uni-AdaFocus-TSM.

These modules do not replace AdaFocus spatial/temporal selection.  They operate
on the selected local features and the full-frame features after selection.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LocalTemporalAdapter(nn.Module):
    """Residual temporal adapter applied to a ResNet layer-3 feature grid.

    Input is the standard flattened video batch [B*T, C, H, W].  Kernel size 1
    is the capacity-matched per-frame control; kernel size 3 mixes time.
    The residual scale starts at zero so a pretrained backbone is unchanged at
    initialization.
    """

    def __init__(self, channels=1024, bottleneck=256, temporal_kernel=3):
        super().__init__()
        if temporal_kernel not in (1, 3):
            raise ValueError("temporal_kernel must be 1 or 3")
        self.norm = nn.GroupNorm(1, channels)
        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
        self.temporal = nn.Conv3d(
            bottleneck,
            bottleneck,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(temporal_kernel // 2, 0, 0),
            groups=bottleneck,
            bias=False,
        )
        self.up = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False)
        self.activation = nn.GELU()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, features, num_segments):
        bt, channels, height, width = features.shape
        if bt % num_segments:
            raise ValueError("flattened batch is not divisible by num_segments")
        batch = bt // num_segments
        residual = features
        features = self.down(self.norm(features))
        bottleneck = features.shape[1]
        features = features.reshape(batch, num_segments, bottleneck, height, width)
        features = features.permute(0, 2, 1, 3, 4).contiguous()
        features = self.activation(self.temporal(features))
        features = features.permute(0, 2, 1, 3, 4).reshape(bt, bottleneck, height, width)
        return residual + self.alpha * self.up(features)


class ContinuousTimeEncoding(nn.Module):
    """Encode normalized source-frame positions without assuming equal lengths."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, positions):
        if positions.ndim != 2:
            raise ValueError("positions must have shape [B, T]")
        return self.net(positions.unsqueeze(-1).float())


class OrderedTemporalPool(nn.Module):
    """Pool a temporal sequence without discarding motion direction.

    Mean pooling is invariant to frame order.  That is a poor match for the
    sweep/reperfusion distinction, where endpoint direction and motion amount
    matter.  This small head keeps the level feature plus signed and absolute
    temporal differences at one- and two-step scales.
    """

    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim * 5, dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _difference_summary(sequence, lag):
        if sequence.shape[1] <= lag:
            zeros = sequence.new_zeros(sequence.shape[0], sequence.shape[2])
            return zeros, zeros
        difference = sequence[:, lag:] - sequence[:, :-lag]
        return difference.mean(dim=1), difference.abs().mean(dim=1)

    def forward(self, sequence):
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [B, T, D]")
        signed_1, magnitude_1 = self._difference_summary(sequence, 1)
        signed_2, magnitude_2 = self._difference_summary(sequence, 2)
        features = torch.cat(
            [
                sequence.mean(dim=1),
                signed_1,
                magnitude_1,
                signed_2,
                magnitude_2,
            ],
            dim=-1,
        )
        return self.norm(self.activation(self.projection(features)))


class LocalContextInteraction(nn.Module):
    """Classify pooled local/global evidence after optional cross-attention.

    ``mode='mlp'`` is the required same-input, same-head control.  In
    ``mode='cross_attention'`` each local spatiotemporal token reads all global
    spatiotemporal tokens.  Real normalized source-frame positions are added to
    each token before attention, so T_local and T_global may differ.
    """

    def __init__(
        self,
        local_channels,
        global_channels,
        dim,
        num_classes,
        mode="cross_attention",
        heads=4,
        dropout=0.1,
        local_grid_size=3,
        global_grid_size=3,
        pooling="mean",
        output_init="gated",
    ):
        super().__init__()
        if mode not in {"mlp", "cross_attention"}:
            raise ValueError("mode must be 'mlp' or 'cross_attention'")
        if dim % heads:
            raise ValueError("interaction dim must be divisible by attention heads")
        if pooling not in {"mean", "ordered_difference"}:
            raise ValueError("pooling must be 'mean' or 'ordered_difference'")
        if output_init not in {"gated", "zero_head"}:
            raise ValueError("output_init must be 'gated' or 'zero_head'")
        self.mode = mode
        self.dim = dim
        self.pooling = pooling
        self.output_init = output_init
        self.local_grid_size = local_grid_size
        self.global_grid_size = global_grid_size
        self.local_projection = nn.Conv2d(local_channels, dim, kernel_size=1, bias=False)
        self.global_projection = nn.Conv2d(global_channels, dim, kernel_size=1, bias=False)
        self.local_time = ContinuousTimeEncoding(dim)
        self.global_time = ContinuousTimeEncoding(dim)
        if mode == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                dim, heads, dropout=dropout, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(dim)
            # The outer correction gate ``gamma`` already guarantees exact
            # baseline equivalence.  Starting beta at one avoids a redundant
            # double-zero gate that can keep attention dormant in short runs.
            self.beta = nn.Parameter(torch.ones(1))
        else:
            self.cross_attention = None
            self.attention_norm = None
            self.register_parameter("beta", None)
        if pooling == "ordered_difference":
            self.local_pool = OrderedTemporalPool(dim)
            self.global_pool = OrderedTemporalPool(dim)
        else:
            self.local_pool = None
            self.global_pool = None
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )
        if output_init == "zero_head":
            # Preserve exact baseline logits while allowing the correction
            # head to receive gradients immediately.  The old zero gamma gate
            # only trained gamma on step one and remained effectively dormant
            # in the formal FSN run (|gamma| < 5e-6).
            nn.init.zeros_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)
            self.gamma = nn.Parameter(torch.ones(1))
        else:
            # Backward-compatible FSN-v1 initialization.
            self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _project_grid(features, projection, grid_size):
        if features.ndim != 5:
            raise ValueError("grid features must have shape [B, T, C, H, W]")
        batch, time, channels, height, width = features.shape
        features = features.reshape(batch * time, channels, height, width)
        features = projection(features)
        features = F.adaptive_avg_pool2d(features, (grid_size, grid_size))
        features = features.reshape(batch, time, -1, grid_size * grid_size)
        return features.permute(0, 1, 3, 2).contiguous()

    @staticmethod
    def _add_time(tokens, time_encoding):
        return tokens + time_encoding.unsqueeze(2)

    def forward(self, local_grid, global_grid, local_positions, global_positions, return_attention=False):
        local = self._project_grid(local_grid, self.local_projection, self.local_grid_size)
        global_ = self._project_grid(global_grid, self.global_projection, self.global_grid_size)
        local = self._add_time(local, self.local_time(local_positions))
        global_ = self._add_time(global_, self.global_time(global_positions))
        batch = local.shape[0]
        local_tokens = local.reshape(batch, -1, self.dim)
        global_tokens = global_.reshape(batch, -1, self.dim)
        attention = None
        if self.cross_attention is not None:
            update, attention = self.cross_attention(
                self.attention_norm(local_tokens), global_tokens, global_tokens,
                need_weights=return_attention,
            )
            local_tokens = local_tokens + self.beta * update
        if self.pooling == "ordered_difference":
            local_sequence = local_tokens.reshape(
                batch, local.shape[1], local.shape[2], self.dim
            ).mean(dim=2)
            global_sequence = global_tokens.reshape(
                batch, global_.shape[1], global_.shape[2], self.dim
            ).mean(dim=2)
            local_pooled = self.local_pool(local_sequence)
            global_pooled = self.global_pool(global_sequence)
        else:
            local_pooled = local_tokens.mean(dim=1)
            global_pooled = global_tokens.mean(dim=1)
        logits = self.gamma * self.classifier(torch.cat([local_pooled, global_pooled], dim=-1))
        return logits, attention
