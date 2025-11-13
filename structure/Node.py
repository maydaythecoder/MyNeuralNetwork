from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass
class Node:
    """
    Container for dense-layer parameters.

    The gradients are stored alongside the parameters to keep modules stateless with respect to
    optimizers; upstream training code collects `(weights, grads)` pairs from each layer.
    """

    weights: FloatArray
    biases: FloatArray
    grad_weights: FloatArray = field(init=False)
    grad_biases: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

    @classmethod
    def initialize(
        cls,
        input_dim: int,
        output_dim: int,
        rng: np.random.Generator | None = None,
    ) -> "Node":
        """
        He initialization keeps variance stable for ReLU-like activations.

        weights ~ N(0, 2 / input_dim)
        biases = 0
        """

        generator = rng or np.random.default_rng()
        limit = np.sqrt(2.0 / float(input_dim))
        weights = generator.normal(loc=0.0, scale=limit, size=(input_dim, output_dim))
        biases = np.zeros((1, output_dim), dtype=np.float64)
        return cls(weights=weights.astype(np.float64), biases=biases)

    def parameter_shapes(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return self.weights.shape, self.biases.shape

    def zero_gradients(self) -> None:
        self.grad_weights.fill(0.0)
        self.grad_biases.fill(0.0)


__all__ = ["Node"]

