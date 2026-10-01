"""Opt-in single-clip global-to-local interaction after local ResNet layer 2.

The aligned candidate adds a fixed distance prior to ordinary content attention.
Positions refer to the selected cache frames, not verified physical timestamps.
Crop coordinates use exactly the affine transform used for image sampling.
No role labels, external boxes, or other clips enter this module.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def crop_affine_theta(action, image_size, patch_size, input_patch_size):
    """Return the Original ``align_corners=False`` crop transform (x then y).

    Original actions store y, x, y-scale, x-scale. Keep its released 224/96
    scale convention, including when cropping seven-cell feature maps.
    """
    theta = torch.zeros((action.size(0), 2, 3), device=action.device)
    patch_scale = action[:, 2:4] * (224 - 96) / input_patch_size + 96 / input_patch_size
    patch_coordinate = action[:, :2] * (image_size - patch_size * patch_scale)
    x1, x2 = patch_coordinate[:, 1], patch_coordinate[:, 1] + patch_size * patch_scale[:, 1]
    y1, y2 = patch_coordinate[:, 0], patch_coordinate[:, 0] + patch_size * patch_scale[:, 0]
    theta[:, 0, 0] = patch_size * patch_scale[:, 1] / image_size
    theta[:, 1, 1] = patch_size * patch_scale[:, 0] / image_size
    theta[:, 0, 2] = -1 + (x1 + x2) / image_size
    theta[:, 1, 2] = -1 + (y1 + y2) / image_size
    return theta


def pooled_grid_centers(height, width, grid, device):
    """Nominal feature-grid centroids of actual adaptive-pooling bins.

    Adaptive bins overlap when input size is not divisible by output size;
    evenly spaced output-grid centers would then be geometrically inaccurate.
    These are feature-grid coordinates, not exact receptive-field boundaries.
    """
    def axis(size):
        index = torch.arange(grid, device=device, dtype=torch.float32)
        left = torch.floor(index * size / grid)
        right = torch.ceil((index + 1) * size / grid)
        return (left + right) / (2 * size)
    y, x = torch.meshgrid(axis(height), axis(width), indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(grid * grid, 2)


class AlignedContextResidual(nn.Module):
    """Content attention, fixed correspondence prior, or local capacity control.

    All three modes have the same trainable parameter count for even global
    channel widths. The capacity control replaces the global projection with
    an active two-layer local MLP of exactly the same size and attends only
    within each local frame. It has no unused global-context parameters.
    """

    def __init__(self, channels=512, dim=64, global_channels=1280, grid=3,
                 mode="aligned", time_scale=0.25, spatial_scale=1.0,
                 image_size=224, patch_size=128):
        super().__init__()
        if mode not in {"plain", "aligned", "capacity"}:
            raise ValueError("context mode must be plain, aligned, or capacity")
        for name, value in (("channels", channels), ("dim", dim),
                            ("global_channels", global_channels), ("grid", grid),
                            ("image_size", image_size), ("patch_size", patch_size)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if global_channels % 2:
            raise ValueError("global_channels must be even for exact capacity matching")
        for name, value in (("time_scale", time_scale), ("spatial_scale", spatial_scale)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.channels, self.dim, self.global_channels, self.grid = channels, dim, global_channels, grid
        self.mode = mode
        self.time_scale, self.spatial_scale = float(time_scale), float(spatial_scale)
        self.image_size, self.patch_size = image_size, patch_size
        self.enabled = True
        self.time_prior_enabled = True
        self.spatial_prior_enabled = True
        self.last_diagnostics = {}
        self.norm = nn.GroupNorm(1, channels)
        self.down = nn.Conv2d(channels, dim, 1, bias=False)
        if mode == "capacity":
            self.global_projection = None
            self.capacity_mlp = nn.Sequential(
                nn.Linear(dim, global_channels // 2, bias=False), nn.GELU(),
                nn.Linear(global_channels // 2, dim, bias=False),
            )
        else:
            self.global_projection = nn.Conv2d(global_channels, dim, 1, bias=False)
            self.capacity_mlp = None
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.up = nn.Conv2d(dim, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    @staticmethod
    def _assert(condition, message):
        assertion = getattr(torch, "_assert_async", torch._assert)
        assertion(condition, message)

    def _positions(self, values, shape, device, name):
        if values is None or values.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if values.device != device:
            raise ValueError(f"{name} must be on the features device")
        self._assert(torch.isfinite(values).all(), f"{name} must be finite")
        self._assert(((values >= 0) & (values <= 1)).all(), f"{name} must lie in [0, 1]")
        self._assert((values[:, 1:] >= values[:, :-1]).all(), f"{name} must be sorted")

    def correspondence_prior(self, positions, global_positions, crop_actions,
                             local_shape, global_shape):
        """Return detached [B,Tlocal*grid²,Tglobal*grid²] attention bias."""
        batch, time_local = positions.shape
        time_global = global_positions.shape[1]
        cells = self.grid * self.grid
        local_xy = pooled_grid_centers(*local_shape, self.grid, positions.device)
        global_xy = pooled_grid_centers(*global_shape, self.grid, positions.device)
        theta = crop_affine_theta(crop_actions.detach().float(), self.image_size,
                                 self.patch_size, self.patch_size)
        local_homogeneous = torch.cat((local_xy * 2 - 1,
                                       local_xy.new_ones(cells, 1)), dim=-1)
        mapped = torch.einsum("bij,nj->bni", theta, local_homogeneous)
        mapped = (mapped + 1) / 2
        prior = positions.new_zeros((batch, time_local, cells, time_global, cells), dtype=torch.float32)
        if self.time_prior_enabled:
            distance = positions.detach().float()[:, :, None] - global_positions.detach().float()[:, None, :]
            prior = prior - 0.5 * (distance[:, :, None, :, None] / self.time_scale).square()
        if self.spatial_prior_enabled:
            distance = mapped[:, :, None, :] - global_xy[None, None, :, :]
            prior = prior - 0.5 * distance.square().sum(-1)[:, None, :, None, :] / self.spatial_scale ** 2
        return prior.reshape(batch, time_local * cells, time_global * cells).detach()

    def diagnostics(self):
        return dict(self.last_diagnostics)

    def forward(self, features, num_segments, global_context=None, positions=None,
                global_positions=None, crop_actions=None):
        if not self.enabled:
            self.last_diagnostics = {}
            return features
        if features.ndim != 4 or features.shape[1] != self.channels or not features.is_floating_point():
            raise ValueError("features must be floating [B*T, channels, H, W]")
        if not isinstance(num_segments, int) or num_segments < 1 or features.shape[0] % num_segments:
            raise ValueError("num_segments must divide the flattened batch")
        bt, _, height, width = features.shape
        if min(bt, height, width) < 1:
            raise ValueError("features must have nonempty dimensions")
        batch = bt // num_segments
        projected = self.down(self.norm(features))
        pooled = F.adaptive_avg_pool2d(projected, (self.grid, self.grid))
        cells = self.grid * self.grid
        local_tokens = pooled.flatten(2).transpose(1, 2)
        prior = None
        if self.mode == "capacity":
            query_tokens = local_tokens
            context_tokens = self.capacity_mlp(local_tokens)
        else:
            if (global_context is None or global_context.ndim != 5
                    or global_context.shape[0] != batch
                    or global_context.shape[2] != self.global_channels
                    or min(global_context.shape[1:]) < 1
                    or global_context.device != features.device
                    or not global_context.is_floating_point()):
                raise ValueError("global_context must be floating [B,Tg,Cg,Hg,Wg] on the features device")
            _, time_global, channels_global, global_height, global_width = global_context.shape
            # Preserve Original's gradient boundary: this residual cannot train
            # the global encoder or the spatial/temporal selection policies.
            global_pooled = F.adaptive_avg_pool2d(
                global_context.detach().reshape(batch * time_global, channels_global, global_height, global_width),
                (self.grid, self.grid),
            )
            context_tokens = self.global_projection(global_pooled).flatten(2).transpose(1, 2)
            context_tokens = context_tokens.reshape(batch, time_global * cells, self.dim)
            query_tokens = local_tokens.reshape(batch, num_segments * cells, self.dim)
            if self.mode == "aligned":
                self._positions(positions, (batch, num_segments), features.device, "positions")
                self._positions(global_positions, (batch, time_global), features.device, "global_positions")
                if crop_actions is None or crop_actions.shape != (batch, 4) or crop_actions.device != features.device:
                    raise ValueError("crop_actions must have shape [B,4] on the features device")
                self._assert(torch.isfinite(crop_actions).all(), "crop actions must be finite")
                self._assert(((crop_actions >= 0) & (crop_actions <= 1)).all(), "crop actions must lie in [0,1]")
                prior = self.correspondence_prior(positions, global_positions, crop_actions,
                                                  (height, width), (global_height, global_width))
        query = self.query(query_tokens)
        key, value = self.key(context_tokens), self.value(context_tokens)
        with torch.autocast(device_type=features.device.type, enabled=False):
            logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(self.dim)
            if prior is not None:
                logits = logits + prior
            attention = logits.softmax(dim=-1)
            attended = torch.matmul(attention, value.float()).to(query.dtype)
        attended = self.output(attended)
        attended = attended.reshape(bt, cells, self.dim).transpose(1, 2)
        attended = attended.reshape(bt, self.dim, self.grid, self.grid)
        residual = self.up(attended)
        residual = F.interpolate(residual, size=(height, width), mode="bilinear", align_corners=False)
        residual = residual.to(features.dtype)
        with torch.no_grad():
            input_rms = features.float().square().mean().sqrt()
            residual_rms = residual.float().square().mean().sqrt()
            self.last_diagnostics = {
                "input_rms": input_rms, "residual_rms": residual_rms,
                "residual_to_input_rms": residual_rms / input_rms.clamp_min(1e-12),
                "attention_entropy": -(attention * attention.clamp_min(1e-12).log()).sum(-1).mean(),
                "attention_max_mean": attention.max(-1).values.mean(),
                "time_prior_enabled": input_rms.new_tensor(float(self.mode == "aligned" and self.time_prior_enabled)),
                "spatial_prior_enabled": input_rms.new_tensor(float(self.mode == "aligned" and self.spatial_prior_enabled)),
            }
            if prior is not None:
                self.last_diagnostics["prior_abs_mean"] = prior.abs().mean()
            if positions is not None:
                gaps = positions[:, 1:].float() - positions[:, :-1].float()
                self.last_diagnostics["normalized_gap_mean"] = gaps.mean() if gaps.numel() else input_rms.new_zeros(())
                self.last_diagnostics["duplicate_position_fraction"] = (gaps == 0).float().mean() if gaps.numel() else input_rms.new_zeros(())
        return features + residual
