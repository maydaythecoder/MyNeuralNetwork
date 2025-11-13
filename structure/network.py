from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray

from logic.Optimizer import Optimizer, SGD
from logic.activationFunction import UnaryActivation, relu, softmax
from logic.lossFunction import LossFunction, cross_entropy
from structure.InputLayer import InputLayer
from structure.Node import Node
from structure.hiddenLayer import HiddenLayer
from structure.outputLayer import OutputLayer


FloatArray = NDArray[np.float64]


@dataclass
class NeuralNetwork:
    """
    Minimal MLP built from the modular layer components.

    The network exposes high-level training utilities (`train_step`, `fit`) while delegating the
    math-heavy pieces to the individual modules for clarity.
    """

    input_layer: InputLayer
    hidden_layers: List[HiddenLayer]
    output_layer: OutputLayer
    loss: LossFunction = field(default=cross_entropy)
    optimizer: Optimizer = field(default_factory=lambda: SGD(learning_rate=0.1))

    def forward(self, inputs: FloatArray) -> FloatArray:
        activations = self.input_layer.forward(inputs)
        for layer in self.hidden_layers:
            activations = layer.forward(activations)
        return self.output_layer.forward(activations)

    def backward(self, predictions: FloatArray, targets: FloatArray) -> None:
        loss_gradient = self.loss.derivative(predictions, targets)
        apply_activation = not (
            self.loss is cross_entropy and self.output_layer.activation is softmax
        )
        downstream = self.output_layer.backward(
            loss_gradient, apply_activation_derivative=apply_activation
        )
        for layer in reversed(self.hidden_layers):
            downstream = layer.backward(downstream)

    def parameter_grad_pairs(self) -> List[tuple[FloatArray, FloatArray]]:
        parameter_pairs: List[tuple[FloatArray, FloatArray]] = []
        for layer in self.hidden_layers:
            parameter_pairs.extend(layer.parameters())
        parameter_pairs.extend(self.output_layer.parameters())
        return parameter_pairs

    def zero_gradients(self) -> None:
        for layer in self.hidden_layers:
            layer.params.zero_gradients()
        self.output_layer.params.zero_gradients()

    def train_step(self, batch_inputs: FloatArray, batch_targets: FloatArray) -> float:
        predictions = self.forward(batch_inputs)
        loss_value = self.loss(predictions, batch_targets)
        self.backward(predictions, batch_targets)
        self.optimizer.step(self.parameter_grad_pairs())
        self.zero_gradients()
        return loss_value

    def fit(
        self,
        inputs: FloatArray,
        targets: FloatArray,
        epochs: int = 100,
        batch_size: int = 32,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> List[float]:
        generator = rng or np.random.default_rng()
        num_samples = inputs.shape[0]
        history: List[float] = []

        for _ in range(epochs):
            indices = np.arange(num_samples)
            if shuffle:
                generator.shuffle(indices)
            batches = [
                indices[start : start + batch_size]
                for start in range(0, num_samples, batch_size)
            ]
            epoch_losses: List[float] = []
            for batch_indices in batches:
                batch_inputs = inputs[batch_indices]
                batch_targets = targets[batch_indices]
                loss_value = self.train_step(batch_inputs, batch_targets)
                epoch_losses.append(loss_value)
            history.append(float(np.mean(epoch_losses)))
        return history

    def predict(self, inputs: FloatArray) -> FloatArray:
        probabilities = self.forward(inputs)
        return self.output_layer.predict(probabilities)


def gradient_check(
    network: NeuralNetwork,
    inputs: FloatArray,
    targets: FloatArray,
    epsilon: float = 1e-5,
) -> float:
    """
    Finite-difference gradient check to spot implementation bugs.

    Returns the maximum absolute difference between analytic and numerical gradients.
    Smaller values (<1e-6) typically indicate correct derivatives.
    """

    predictions = network.forward(inputs)
    network.backward(predictions, targets)

    parameter_pairs = network.parameter_grad_pairs()
    analytic_grads = [grad.copy() for _, grad in parameter_pairs]
    params = [param for param, _ in parameter_pairs]

    max_diff = 0.0
    for param, analytic in zip(params, analytic_grads):
        iterator = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
        while not iterator.finished:
            idx = iterator.multi_index
            original = param[idx]

            param[idx] = original + epsilon
            plus_loss = network.loss(network.forward(inputs), targets)

            param[idx] = original - epsilon
            minus_loss = network.loss(network.forward(inputs), targets)

            param[idx] = original
            numerical = (plus_loss - minus_loss) / (2.0 * epsilon)

            diff = float(abs(numerical - analytic[idx]))
            if diff > max_diff:
                max_diff = diff
            iterator.iternext()

    network.zero_gradients()
    return max_diff


def make_toy_classification_dataset(
    samples_per_class: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[FloatArray, FloatArray]:
    """
    Two moons-style synthetic dataset for quick experimentation.

    Generates two interleaving semicircles and returns one-hot encoded targets.
    """

    generator = rng or np.random.default_rng(0)
    angles = generator.uniform(0.0, np.pi, size=samples_per_class)
    radius = 1.0 + 0.1 * generator.normal(size=samples_per_class)
    class_a = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    angles_b = angles + np.pi
    radius_b = 1.0 + 0.1 * generator.normal(size=samples_per_class)
    class_b = np.stack(
        [
            radius_b * np.cos(angles_b) + 0.5,
            radius_b * np.sin(angles_b) - 0.2,
        ],
        axis=1,
    )

    inputs = np.vstack([class_a, class_b]).astype(np.float64)
    labels = np.concatenate(
        [np.zeros(samples_per_class, dtype=np.int64), np.ones(samples_per_class, dtype=np.int64)]
    )
    targets = np.eye(2, dtype=np.float64)[labels]
    return inputs, targets


def build_demo_network(
    input_dim: int,
    hidden_units: Sequence[int],
    output_dim: int,
    activation: UnaryActivation = relu,
    rng: np.random.Generator | None = None,
) -> NeuralNetwork:
    """
    Convenience factory wiring the modular pieces together.

    `hidden_units` allows stacking multiple hidden layers without changing the surrounding code.
    """

    generator = rng or np.random.default_rng(1)
    input_layer = InputLayer(input_dim=input_dim)

    hidden_layers: List[HiddenLayer] = []
    prev_units = input_dim
    for units in hidden_units:
        params = Node.initialize(prev_units, units, rng=generator)
        hidden_layers.append(HiddenLayer(params=params, activation=activation))
        prev_units = units

    output_params = Node.initialize(prev_units, output_dim, rng=generator)
    output_layer = OutputLayer(params=output_params, activation=softmax)

    return NeuralNetwork(
        input_layer=input_layer,
        hidden_layers=hidden_layers,
        output_layer=output_layer,
        loss=cross_entropy,
        optimizer=SGD(learning_rate=0.1, max_grad_norm=1.0),
    )


def demo_training_epoch() -> None:
    """
    Example usage training on a synthetic dataset.

    Prints epoch-level losses and performs a gradient check on the initial parameters for sanity.
    """

    inputs, targets = make_toy_classification_dataset(samples_per_class=200)
    network = build_demo_network(input_dim=inputs.shape[1], hidden_units=(16, 16), output_dim=2)

    initial_diff = gradient_check(network, inputs[:5], targets[:5])
    print(f"Gradient check (max diff): {initial_diff:.6e}")

    losses = network.fit(inputs, targets, epochs=50, batch_size=32)
    print(f"Final training loss: {losses[-1]:.4f}")

    predictions = network.predict(inputs)
    accuracy = np.mean(predictions.flatten() == np.argmax(targets, axis=1))
    print(f"Training accuracy: {accuracy:.3f}")


__all__ = [
    "NeuralNetwork",
    "gradient_check",
    "make_toy_classification_dataset",
    "build_demo_network",
    "demo_training_epoch",
]

