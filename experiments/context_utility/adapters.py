"""Explicit wrappers for an available, audited historical TCN.

This file does not invent a three-block legacy architecture. A real factory must
instantiate the actual historical model, then declare its input/output layouts.
"""

import torch
from torch import Tensor, nn


class LegacyPredictionAdapter(nn.Module):
    """Adapt a true legacy model to ``forward(features[T,F], logits[T,7])``.

    The model receives concatenated frozen visual features and either raw A
    logits or A probabilities, exactly as explicitly declared. Its result is
    merely rearranged to [T,7]; no residual or calibration is silently added.
    A factory requiring other normalization, inputs, outputs, or sequence state
    must implement those genuine semantics itself instead of using this helper.
    """

    def __init__(self, model, *, prediction_input, input_layout, output_layout,
                 singleton="delegate"):
        super().__init__()
        if not isinstance(model, nn.Module):
            raise ValueError("legacy model must be an actual nn.Module")
        if prediction_input not in {"logits", "probabilities"}:
            raise ValueError("prediction_input must explicitly be logits or probabilities")
        if input_layout not in {"BCT", "BTC", "TC", "CT"}:
            raise ValueError("unsupported explicit input_layout")
        if output_layout not in {"BCT", "BTC", "TC", "CT"}:
            raise ValueError("unsupported explicit output_layout")
        if singleton not in {"delegate", "a_logits"}:
            raise ValueError("singleton must explicitly be delegate or a_logits")
        self.model = model
        self.prediction_input = prediction_input
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.singleton = singleton

    def forward(self, features: Tensor, a_logits: Tensor) -> Tensor:
        if (not isinstance(features, Tensor) or not isinstance(a_logits, Tensor)
                or features.ndim != 2 or a_logits.ndim != 2
                or features.shape[0] == 0 or features.shape[1] == 0
                or a_logits.shape != (features.shape[0], 7)):
            raise ValueError("expected features[T,F] and A logits[T,7]")
        if (not features.is_floating_point() or not a_logits.is_floating_point()
                or features.device != a_logits.device or features.dtype != a_logits.dtype
                or not torch.isfinite(features).all() or not torch.isfinite(a_logits).all()):
            raise ValueError("inputs must have one floating dtype/device and finite values")
        if features.shape[0] == 1 and self.singleton == "a_logits":
            return a_logits
        predictions = a_logits if self.prediction_input == "logits" else a_logits.softmax(-1)
        values = torch.cat((features, predictions), dim=-1)
        if self.input_layout == "BCT":
            values = values.T.unsqueeze(0)
        elif self.input_layout == "BTC":
            values = values.unsqueeze(0)
        elif self.input_layout == "CT":
            values = values.T
        output = self.model(values)
        if not isinstance(output, Tensor):
            raise ValueError("legacy model must return a tensor; implement a real custom factory otherwise")
        if self.output_layout == "BCT":
            if output.ndim != 3 or output.shape[0] != 1:
                raise ValueError("legacy output does not match BCT")
            output = output[0].T
        elif self.output_layout == "BTC":
            if output.ndim != 3 or output.shape[0] != 1:
                raise ValueError("legacy output does not match BTC")
            output = output[0]
        elif self.output_layout == "CT":
            output = output.T
        if (output.shape != a_logits.shape or output.device != a_logits.device
                or not output.is_floating_point() or not torch.isfinite(output).all()):
            raise ValueError("legacy output must be finite logits[T,7] on the input device")
        return output
