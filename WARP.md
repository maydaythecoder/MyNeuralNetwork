# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

A modular, educational implementation of a Multi-Layer Perceptron (MLP) neural network built from scratch using NumPy. The architecture emphasizes clean separation of concerns with Protocol-based interfaces and stateless components.

## Architecture

### Core Design Philosophy

The codebase follows a compositional pattern where each component is independent and communicates through well-defined interfaces:

- **Stateless layers**: Layers cache intermediate values during forward passes only for their own backward pass
- **Parameter ownership**: `Node` class owns weights, biases, and their gradients
- **Protocol-based**: `Optimizer`, `Activation`, and `Loss` use Protocol classes for type safety without inheritance
- **Modular composition**: Network construction via factory functions rather than inheritance hierarchies

### Directory Structure

- `structure/`: Layer components and network architecture
  - `Node.py`: Parameter container with He initialization
  - `InputLayer.py`: Input validation layer (no parameters)
  - `hiddenLayer.py`: Dense layer + activation
  - `outputLayer.py`: Final readout layer with prediction utilities
  - `network.py`: Top-level `NeuralNetwork` class that orchestrates training

- `logic/`: Mathematical primitives
  - `activationFunction.py`: Sigmoid, ReLU, Softmax with derivatives
  - `lossFunction.py`: MSE, Cross-Entropy with gradients
  - `Optimizer.py`: SGD with gradient clipping

### Key Patterns

**Gradient Flow**: The network uses explicit backward passes where each layer:

1. Receives upstream gradient
2. Computes local parameter gradients (stored in `Node.grad_weights`, `Node.grad_biases`)
3. Returns downstream gradient for previous layer

**Optimization Decoupling**: Layers return `(parameter, gradient)` tuples via `.parameters()` method. The optimizer receives an iterable of these pairs and mutates parameters in-place, keeping layer code optimizer-agnostic.

**Softmax + Cross-Entropy Fusion**: When using softmax activation with cross-entropy loss, the network automatically skips the softmax derivative computation (collapsed to `predictions - targets`), controlled by `apply_activation_derivative=False` flag in `OutputLayer.backward()`.

## Development Commands

### Setup

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running Demo

```bash
# Run the built-in training demonstration
python3 -c "from structure.network import demo_training_epoch; demo_training_epoch()"
```

### Interactive Testing

```python
# Example: Build and train a custom network
from structure.network import build_demo_network, make_toy_classification_dataset
import numpy as np

# Create dataset
inputs, targets = make_toy_classification_dataset(samples_per_class=200)

# Build network: 2 inputs -> [16, 16] hidden -> 2 outputs
network = build_demo_network(
    input_dim=2,
    hidden_units=(16, 16),
    output_dim=2,
    rng=np.random.default_rng(42)
)

# Train
losses = network.fit(inputs, targets, epochs=50, batch_size=32)

# Evaluate
predictions = network.predict(inputs)
accuracy = np.mean(predictions.flatten() == np.argmax(targets, axis=1))
```

### Gradient Checking

```python
# Verify gradient implementations with finite differences
from structure.network import gradient_check, build_demo_network, make_toy_classification_dataset

inputs, targets = make_toy_classification_dataset(samples_per_class=50)
network = build_demo_network(input_dim=2, hidden_units=(8,), output_dim=2)

max_diff = gradient_check(network, inputs[:5], targets[:5])
print(f"Max gradient error: {max_diff:.6e}")  # Should be < 1e-6
```

## Type Annotations

The codebase uses modern Python type hints:

- `FloatArray = NDArray[np.float64]` for all array operations
- `from __future__ import annotations` for forward references
- Protocol classes for structural subtyping

When adding new components, maintain this pattern and ensure compatibility with `np.float64` dtype.

## Extending the Network

### Adding New Activation Functions

Create a `UnaryActivation` instance in `logic/activationFunction.py`:

```python
def _custom_forward(inputs: FloatArray) -> FloatArray:
    return ...  # Your activation

def _custom_backward(activated_inputs: FloatArray) -> FloatArray:
    return ...  # Derivative w.r.t. activated output

custom_activation = UnaryActivation(
    forward=_custom_forward,
    backward=_custom_backward
)
```

### Adding New Optimizers

Implement the `Optimizer` Protocol in `logic/Optimizer.py`:

```python
@dataclass
class Adam:
    learning_rate: float = 0.001
    # ... momentum parameters

    def step(self, params_and_grads: Iterable[ParamGrad]) -> None:
        for parameters, gradients in params_and_grads:
            # Update logic here
            parameters -= ...
```

### Adding New Loss Functions

Create a `LossFunction` instance in `logic/lossFunction.py`:

```python
def _custom_loss(predictions: FloatArray, targets: FloatArray) -> float:
    return ...

def _custom_loss_grad(predictions: FloatArray, targets: FloatArray) -> FloatArray:
    return ...

custom_loss = LossFunction(
    forward=_custom_loss,
    backward=_custom_loss_grad
)
```

## Important Implementation Details

- **Batch-averaged gradients**: All gradient computations divide by batch size, so optimizers don't need to account for it
- **Numerical stability**: Softmax and cross-entropy use log-space tricks and clipping to prevent overflow
- **He initialization**: Used by default in `Node.initialize()` for ReLU-compatible variance scaling
- **No automatic differentiation**: All derivatives are hand-coded for educational transparency

## Dependencies

- `numpy>=1.26,<2.0`: Only external dependency
- Python 3.10+ required for modern type hints (`X | Y` union syntax)
