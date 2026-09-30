"""Early, opt-in local evidence for the existing Uni-AdaFocus local encoder.

The matching branch is a MotionSqueeze-inspired research prototype, not a
reproduction of that paper or an optical-flow estimator. It compares adjacent
*selected* frames inside each clip; their normalized source positions are
diagnostics, not a physical clock. No clip-order or surgical-role prior is used.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class LocalMotionEvidence(nn.Module):
    """Inject local correspondence or a capacity-matched appearance control.

    ``features`` uses the backbone's flattened ``[B*T, C, H, W]`` layout. The
    global option uses one clip-level condition: temporal averaging retains a
    learned, flattened spatial grid, rather than averaging away its cells.
    It does not pretend that the global and local frame arrays are aligned.

    The final residual projection starts at zero, giving exact initial identity
    and an immediately trainable output projection, with no outer zero gate.
    ``enabled=False`` bypasses the branch, including all branch computation.
    Diagnostics remain detached tensors on the input device; callers decide
    when to synchronize them for logging.
    """

    def __init__(
        self,
        channels=512,
        bottleneck=64,
        mode="matching",
        window_size=3,
        temperature=0.07,
        context_mode="none",
        context_channels=1280,
        context_grid=2,
    ):
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        if not isinstance(bottleneck, int) or bottleneck < 3:
            raise ValueError("bottleneck must be an integer >= 3")
        if mode not in {"matching", "appearance"}:
            raise ValueError("mode must be 'matching' or 'appearance'")
        if not isinstance(window_size, int) or window_size < 3 or window_size % 2 != 1:
            raise ValueError("window_size must be an odd integer >= 3")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if context_mode not in {"none", "global"}:
            raise ValueError("context_mode must be 'none' or 'global'")
        if not isinstance(context_channels, int) or context_channels < 1:
            raise ValueError("context_channels must be a positive integer")
        if not isinstance(context_grid, int) or context_grid < 1:
            raise ValueError("context_grid must be a positive integer")

        self.channels = channels
        self.bottleneck = bottleneck
        self.mode = mode
        self.window_size = window_size
        self.temperature = float(temperature)
        self.context_mode = context_mode
        self.context_channels = context_channels
        self.context_grid = context_grid
        self.enabled = True
        self.context_enabled = True
        self.last_diagnostics = {}

        self.norm = nn.GroupNorm(1, channels)
        self.down = nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.evidence = nn.Sequential(
            nn.Conv2d(3, bottleneck, 3, padding=1, bias=False),
            nn.GroupNorm(1, bottleneck),
            nn.GELU(),
        )
        self.up = nn.Conv2d(bottleneck, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

        if context_mode == "global":
            self.context_projection = nn.Conv2d(
                context_channels, bottleneck, 1, bias=False
            )
            self.context_global = nn.Linear(
                bottleneck * context_grid * context_grid, bottleneck
            )
            self.context_local = nn.Conv2d(bottleneck, bottleneck, 1)
            self.context_gate = nn.Conv2d(bottleneck, bottleneck, 1)
            nn.init.zeros_(self.context_gate.weight)
            nn.init.zeros_(self.context_gate.bias)
        else:
            # No unused trainable context parameters in the local-only control.
            self.context_projection = None
            self.context_global = None
            self.context_local = None
            self.context_gate = None

    @staticmethod
    def _numeric_assert(condition, message):
        # GPU scalar .item()/bool() would synchronize every forward. Newer
        # PyTorch provides an asynchronous device assertion for this purpose.
        asynchronous_assert = getattr(torch, "_assert_async", None)
        if asynchronous_assert is not None:
            asynchronous_assert(condition, message)
        else:
            torch._assert(condition, message)

    def _validate(self, features, num_segments, global_context, positions):
        if features.ndim != 4:
            raise ValueError("features must have shape [B*T, C, H, W]")
        if not features.is_floating_point():
            raise ValueError("features must be floating-point")
        if not isinstance(num_segments, int) or num_segments < 1:
            raise ValueError("num_segments must be a positive integer")
        bt, channels, height, width = features.shape
        if bt < 1 or bt % num_segments:
            raise ValueError("flattened batch must be positive and divisible by num_segments")
        if channels != self.channels or height < 1 or width < 1:
            raise ValueError("features have incompatible channels or empty spatial dimensions")
        batch = bt // num_segments
        if positions is not None:
            if positions.shape != (batch, num_segments):
                raise ValueError("positions must have shape [B, T]")
            if positions.device != features.device:
                raise ValueError("positions and features must be on the same device")
            self._numeric_assert(torch.isfinite(positions).all(), "positions must be finite")
            self._numeric_assert(
                ((positions >= 0) & (positions <= 1)).all(),
                "positions must be normalized to [0, 1]",
            )
            self._numeric_assert(
                (positions[:, 1:] >= positions[:, :-1]).all(),
                "positions must be sorted within each clip (duplicates are allowed)",
            )
        if self.context_mode == "global" and self.context_enabled:
            if global_context is None or global_context.ndim != 5:
                raise ValueError("global context must have shape [B, Tg, Cg, Hg, Wg]")
            if (
                global_context.shape[0] != batch
                or global_context.shape[1] < 1
                or global_context.shape[2] != self.context_channels
                or min(global_context.shape[3:]) < 1
            ):
                raise ValueError("global context has incompatible shape")
            if global_context.device != features.device:
                raise ValueError("global context and features must be on the same device")
            if not global_context.is_floating_point():
                raise ValueError("global context must be floating-point")
        return batch, height, width

    def diagnostics(self):
        """Return a copy of detached device scalars from the latest forward."""
        return dict(self.last_diagnostics)

    def estimate_motion(self, projected):
        """Return ``[B,T,3,H,W]``: normalized dx, dy, and concentration.

        A positive dx means that a source feature matches a feature to its right
        in the next selected frame. The last frame has no forward observation
        and receives zeros. Confidence measures distribution concentration;
        it is not a calibrated probability that motion is correct. Correlations
        and softmax stay float32 even under autocast, with invalid borders masked.
        This public method also permits isolated correspondence diagnostics.
        """
        if projected.ndim != 5 or projected.shape[2] != self.bottleneck:
            raise ValueError("projected features must have shape [B, T, bottleneck, H, W]")
        batch, time, channels, height, width = projected.shape
        if min(batch, time, height, width) < 1:
            raise ValueError("projected features must have nonempty dimensions")
        with torch.autocast(device_type=projected.device.type, enabled=False):
            projected = F.normalize(projected.float(), dim=2, eps=1e-6)
            if time == 1:
                return projected.new_zeros(batch, time, 3, height, width)
            source = projected[:, :-1].reshape(-1, channels, height, width)
            target = projected[:, 1:].reshape(-1, channels, height, width)
            radius = self.window_size // 2
            padded = F.pad(target, (radius, radius, radius, radius))
            rows = torch.arange(height, device=projected.device).view(height, 1)
            cols = torch.arange(width, device=projected.device).view(1, width)
            scores, masks, offsets = [], [], []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    neighbor = padded[
                        :, :, radius + dy : radius + dy + height,
                        radius + dx : radius + dx + width,
                    ]
                    scores.append((source * neighbor).sum(dim=1))
                    masks.append(
                        (rows + dy >= 0) & (rows + dy < height)
                        & (cols + dx >= 0) & (cols + dx < width)
                    )
                    offsets.append((dx / radius, dy / radius))
            logits = torch.stack(scores, dim=1) / self.temperature
            valid = torch.stack(masks, dim=0).unsqueeze(0)
            probabilities = logits.masked_fill(~valid, float("-inf")).softmax(dim=1)
            offsets = projected.new_tensor(offsets)
            dx = (probabilities * offsets[:, 0].view(1, -1, 1, 1)).sum(dim=1)
            dy = (probabilities * offsets[:, 1].view(1, -1, 1, 1)).sum(dim=1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
            valid_count = valid.sum(dim=1).to(projected.dtype)
            concentration = torch.where(
                valid_count > 1,
                1 - entropy / valid_count.clamp_min(2).log(),
                torch.zeros_like(entropy),
            ).clamp(0, 1)
            motion = torch.stack((dx, dy, concentration), dim=1)
            motion = motion.reshape(batch, time - 1, 3, height, width)
            return torch.cat((motion, motion.new_zeros(batch, 1, 3, height, width)), dim=1)

    def _calibration(self, projected, global_context, batch, num_segments):
        batch_global, time_global, channels, height, width = global_context.shape
        summary = F.adaptive_avg_pool2d(
            global_context.reshape(batch_global * time_global, channels, height, width),
            (self.context_grid, self.context_grid),
        )
        summary = self.context_projection(summary)
        summary = summary.reshape(batch, time_global, -1).mean(dim=1)
        summary = self.context_global(summary)
        summary = summary[:, None].expand(batch, num_segments, -1)
        summary = summary.reshape(batch * num_segments, self.bottleneck, 1, 1)
        combined = F.gelu(self.context_local(projected) + summary)
        return 1 + self.context_gate(combined).tanh()

    def forward(self, features, num_segments, global_context=None, positions=None):
        if not self.enabled:
            self.last_diagnostics = {}
            return features
        batch, height, width = self._validate(
            features, num_segments, global_context, positions
        )
        projected = self.down(self.norm(features))
        if self.mode == "matching":
            observation = self.estimate_motion(
                projected.reshape(batch, num_segments, self.bottleneck, height, width)
            ).reshape(batch * num_segments, 3, height, width)
        else:
            # Every projected channel contributes, without adding parameters to
            # the appearance control or borrowing information from other frames.
            observation = torch.cat(
                [chunk.mean(dim=1, keepdim=True) for chunk in torch.tensor_split(projected, 3, dim=1)],
                dim=1,
            )
        evidence = self.evidence(observation.to(projected.dtype))
        calibration = None
        if self.context_mode == "global" and self.context_enabled:
            calibration = self._calibration(projected, global_context, batch, num_segments)
            evidence = evidence * calibration
        residual = self.up(evidence).to(features.dtype)

        with torch.no_grad():
            input_rms = features.float().square().mean().sqrt()
            residual_rms = residual.float().square().mean().sqrt()
            diagnostics = {
                "input_rms": input_rms,
                "residual_rms": residual_rms,
                "residual_to_input_rms": residual_rms / input_rms.clamp_min(1e-12),
                "observation_rms": observation.float().square().mean().sqrt(),
            }
            if self.mode == "matching":
                nonterminal = observation.reshape(batch, num_segments, 3, height, width)[:, :-1, 2]
                diagnostics["matching_confidence"] = (
                    nonterminal.mean() if num_segments > 1 else input_rms.new_zeros(())
                )
            if calibration is not None:
                diagnostics["calibration_delta_rms"] = (calibration.float() - 1).square().mean().sqrt()
            if positions is not None:
                gaps = positions[:, 1:].float() - positions[:, :-1].float()
                diagnostics["normalized_gap_mean"] = gaps.mean() if num_segments > 1 else input_rms.new_zeros(())
                diagnostics["normalized_gap_max"] = gaps.max() if num_segments > 1 else input_rms.new_zeros(())
                diagnostics["duplicate_position_fraction"] = (
                    (gaps == 0).float().mean() if num_segments > 1 else input_rms.new_zeros(())
                )
            self.last_diagnostics = diagnostics
        return features + residual
