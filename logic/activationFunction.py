from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class Activation(Protocol):
    """Interface for activation primitives providing forward and derivative calls."""

    def __call__(self, inputs: FloatArray) -> FloatArray:
        raise NotImplementedError

    def derivative(self, activated_inputs: FloatArray) -> FloatArray:
        raise NotImplementedError


@dataclass(frozen=True)
class UnaryActivation:
    """
    Helper to keep forward and derivative logic together.

    Storing `forward` and `derivative` enables re-use across layers without coupling to layer
    internals.
    """

    forward: Callable[[FloatArray], FloatArray]
    backward: Callable[[FloatArray], FloatArray]

    def __call__(self, inputs: FloatArray) -> FloatArray:
        return self.forward(inputs)

    def derivative(self, activated_inputs: FloatArray) -> FloatArray:
        return self.backward(activated_inputs)


def _sigmoid_forward(inputs: FloatArray) -> FloatArray:
    # SAFETY: Clip avoids overflow when exponentiating large magnitude inputs.
    clipped = np.clip(inputs, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _sigmoid_backward(activated_inputs: FloatArray) -> FloatArray:
    return activated_inputs * (1.0 - activated_inputs)


sigmoid: UnaryActivation = UnaryActivation(
    forward=_sigmoid_forward,
    backward=_sigmoid_backward,
)


def _relu_forward(inputs: FloatArray) -> FloatArray:
    return np.maximum(0.0, inputs)


def _relu_backward(activated_inputs: FloatArray) -> FloatArray:
    return (activated_inputs > 0.0).astype(np.float64)


relu: UnaryActivation = UnaryActivation(
    forward=_relu_forward,
    backward=_relu_backward,
)


def _softmax_forward(inputs: FloatArray) -> FloatArray:
    shifted = inputs - np.max(inputs, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _softmax_backward(activated_inputs: FloatArray) -> FloatArray:
    """
    Gradient of the softmax output with respect to its inputs.

    Returns Jacobian diagonals flattened to match upstream gradient expectations.
    Each row corresponds to the diagonal entries for a sample's softmax vector, which is
    sufficient when paired with cross-entropy loss (full Jacobian rarely needed).
    """
    return activated_inputs * (1.0 - activated_inputs)


softmax: UnaryActivation = UnaryActivation(
    forward=_softmax_forward,
    backward=_softmax_backward,
)


__all__ = [
    "Activation",
    "UnaryActivation",
    "sigmoid",
    "relu",
    "softmax",
]

