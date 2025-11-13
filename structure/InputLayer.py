from __future__ import annotations

from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np


FloatArray = NDArray[np.float64]


@dataclass
class InputLayer:
    """
    No learnable parameters; validates incoming feature dimensions.

    Forward simply caches the batch for downstream layers. Backward passes the gradient upstream
    unchanged.
    """

    input_dim: int
    last_output: FloatArray | None = None

    def forward(self, inputs: FloatArray) -> FloatArray:
        if inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, received {inputs.shape[1]}"
            )
        self.last_output = inputs
        return inputs

    def backward(self, upstream_gradient: FloatArray) -> FloatArray:
        # SAFETY: Input layer has no parameters, so upstream gradient flows through untouched.
        return upstream_gradient


__all__ = ["InputLayer"]

