from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class Loss(Protocol):
    """Interface for loss primitives mirroring activation structure."""

    def __call__(self, predictions: FloatArray, targets: FloatArray) -> float:
        raise NotImplementedError

    def derivative(self, predictions: FloatArray, targets: FloatArray) -> FloatArray:
        raise NotImplementedError


@dataclass(frozen=True)
class LossFunction:
    forward: Callable[[FloatArray, FloatArray], float]
    backward: Callable[[FloatArray, FloatArray], FloatArray]

    def __call__(self, predictions: FloatArray, targets: FloatArray) -> float:
        return float(self.forward(predictions, targets))

    def derivative(self, predictions: FloatArray, targets: FloatArray) -> FloatArray:
        return self.backward(predictions, targets)


def _mean_squared_error(predictions: FloatArray, targets: FloatArray) -> float:
    residual = predictions - targets
    return float(np.mean(np.square(residual)))


def _mean_squared_error_grad(predictions: FloatArray, targets: FloatArray) -> FloatArray:
    batch_size = predictions.shape[0]
    return (2.0 / batch_size) * (predictions - targets)


mean_squared_error = LossFunction(
    forward=_mean_squared_error,
    backward=_mean_squared_error_grad,
)


def _cross_entropy(predictions: FloatArray, targets: FloatArray) -> float:
    epsilon = 1e-12
    stabilized = np.clip(predictions, epsilon, 1.0 - epsilon)
    losses = -np.sum(targets * np.log(stabilized), axis=1)
    return float(np.mean(losses))


def _cross_entropy_grad(predictions: FloatArray, targets: FloatArray) -> FloatArray:
    batch_size = predictions.shape[0]
    epsilon = 1e-12
    stabilized = np.clip(predictions, epsilon, 1.0 - epsilon)
    return (stabilized - targets) / batch_size


cross_entropy = LossFunction(
    forward=_cross_entropy,
    backward=_cross_entropy_grad,
)


__all__ = [
    "Loss",
    "LossFunction",
    "mean_squared_error",
    "cross_entropy",
]

