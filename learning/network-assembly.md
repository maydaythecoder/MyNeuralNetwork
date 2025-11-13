# Network Assembly and Training Flow

## Objective

Bring together all previously covered components to form a working neural network you can train, evaluate, and debug.

### Core Classes and Functions

- `NeuralNetwork` (in `structure/network.py`)
- `gradient_check`
- `make_toy_classification_dataset`
- `build_demo_network`
- `demo_training_epoch`

### Forward Pass Recap

1. **Input layer** validates feature dimensions.
2. **Hidden layers** transform activations sequentially.
3. **Output layer** produces probabilities or scores.

All layers reuse the activation, loss, optimizer, and node components you have already studied.

### Backward Pass and Parameter Updates

1. Compute `loss_gradient = loss.derivative(predictions, targets)`.
2. Output layer handles the gradient, optionally skipping its derivative when paired with softmax + cross entropy.
3. Hidden layers run backward in reverse order, each producing gradients for its parameters and the upstream layer.
4. `parameter_grad_pairs` collects `(weights, grad_weights)` and `(biases, grad_biases)` for every learnable layer.
5. Optimizer `step` mutates parameters, then `zero_gradients` resets buffers for the next iteration.

### Training Utilities

- `train_step`  
  Executes forward pass, computes loss, performs backpropagation, updates parameters, returns the scalar loss.
- `fit`  
  Handles multiple epochs, batch shuffling, and tracks average loss per epoch.
- `predict`  
  Runs a forward pass and delegates to the output layer’s `predict` to obtain class labels.

### Quality Checks and Helpers

- `gradient_check`  
  Performs finite-difference checks to compare analytical and numerical gradients. Use it on small batches (e.g., 5 samples) to validate new layers.
- `make_toy_classification_dataset`  
  Generates a “two moons” dataset with one-hot targets, perfect for experimentation without external data.
- `build_demo_network`  
  Factory that assembles the full network with configurable hidden units and activations.
- `demo_training_epoch`  
  Demonstrates end-to-end usage: runs gradient checking, trains on the toy dataset, prints loss and accuracy.

### Suggested Next Experiments

- Change `hidden_units` in `build_demo_network` to see how depth and width affect learning.
- Swap the activation passed to `build_demo_network` (e.g., sigmoid) and watch training behaviour.
- Adjust `learning_rate` or enable `max_grad_norm` in `SGD` to practice tuning optimization.

### Linking Backward

If any component feels unclear, revisit its dedicated document. The network file wires them together but does not hide their internals—use this transparency to iterate confidently.
