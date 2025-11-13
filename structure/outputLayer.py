from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from logic.activationFunction import UnaryActivation, softmax
from structure.Node import Node


FloatArray = NDArray[np.float64]


@dataclass
class OutputLayer:
    """
    Dense readout layer with configurable activation.

    Default activation is softmax for multi-class classification. Swap in identity or sigmoid when
    doing regression or binary outputs.
    """

    params: Node
    activation: UnaryActivation = softmax
    last_input: FloatArray | None = field(default=None, init=False)
    last_logits: FloatArray | None = field(default=None, init=False)
    last_output: FloatArray | None = field(default=None, init=False)

    def forward(self, inputs: FloatArray) -> FloatArray:
        self.last_input = inputs
        logits = inputs @ self.params.weights + self.params.biases
        self.last_logits = logits
        output = self.activation(logits)
        self.last_output = output
        return output

    def backward(
        self,
        upstream_gradient: FloatArray,
        *,
        apply_activation_derivative: bool = True,
    ) -> FloatArray:
        if self.last_input is None or self.last_output is None:
            raise RuntimeError("forward must be called before backward.")

        batch_size = upstream_gradient.shape[0]
        if apply_activation_derivative:
            activation_prime = self.activation.derivative(self.last_output)
            delta = upstream_gradient * activation_prime
        else:
            delta = upstream_gradient

        self.params.grad_weights = (self.last_input.T @ delta) / batch_size
        self.params.grad_biases = np.sum(delta, axis=0, keepdims=True) / batch_size

        return delta @ self.params.weights.T

    def parameters(self) -> list[tuple[FloatArray, FloatArray]]:
        return [
            (self.params.weights, self.params.grad_weights),
            (self.params.biases, self.params.grad_biases),
        ]

    def predict(self, outputs: FloatArray | None = None) -> FloatArray:
        scores = outputs if outputs is not None else self.last_output
        if scores is None:
            raise RuntimeError("No cached outputs to derive predictions from.")
        return np.argmax(scores, axis=1, keepdims=True)


__all__ = ["OutputLayer"]

