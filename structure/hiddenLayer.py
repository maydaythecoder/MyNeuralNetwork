from __future__ import annotationsf
from dataclasses import dataclass, field
from numpy.typing import NDArray

from logic.activationFunction import UnaryActivation
from structure.Node import Node
import numpy as np


FloatArray = NDArray[np.float64]


@dataclass
class HiddenLayer:
    """
    Dense layer with activation.

    `Node` carries the trainable parameters, enabling the optimizer module to receive a unified view
    of `(parameters, gradients)` pairs across layers.
    """

    params: Node
    activation: UnaryActivation
    last_input: FloatArray | None = field(default=None, init=False)
    last_linear_output: FloatArray | None = field(default=None, init=False)
    last_output: FloatArray | None = field(default=None, init=False)

    def forward(self, inputs: FloatArray) -> FloatArray:
        self.last_input = inputs
        linear_output = inputs @ self.params.weights + self.params.biases
        self.last_linear_output = linear_output
        activated = self.activation(linear_output)
        self.last_output = activated
        return activated

    def backward(self, upstream_gradient: FloatArray) -> FloatArray:
        if self.last_input is None or self.last_output is None:
            raise RuntimeError("forward must be called before backward.")

        batch_size = upstream_gradient.shape[0]
        activation_prime = self.activation.derivative(self.last_output)
        delta = upstream_gradient * activation_prime

        self.params.grad_weights = (self.last_input.T @ delta) / batch_size
        self.params.grad_biases = np.sum(delta, axis=0, keepdims=True) / batch_size

        return delta @ self.params.weights.T

    def parameters(self) -> list[tuple[FloatArray, FloatArray]]:
        return [
            (self.params.weights, self.params.grad_weights),
            (self.params.biases, self.params.grad_biases),
        ]


__all__ = ["HiddenLayer"]

