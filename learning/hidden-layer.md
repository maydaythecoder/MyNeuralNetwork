# Hidden Layer

## Why It Matters

Hidden layers create the non-linear transformations that let the network learn complex patterns. Each hidden layer combines a matrix multiply, a bias addition, and an activation function.

### Step-by-Step Forward Pass

1. **Cache inputs**  
   `self.last_input = inputs` keeps a copy for use during backpropagation.
2. **Linear combination**  
   Compute `linear_output = inputs @ weights + biases`.  
   - `inputs`: shape `(batch_size, input_dim)`  
   - `weights`: shape `(input_dim, output_dim)`  
   - `biases`: shape `(1, output_dim)` broadcast across the batch.
3. **Activation**  
   Apply the configured `UnaryActivation` (e.g., ReLU). The result is stored in `self.last_output`.

### Step-by-Step Backward Pass

1. **Safety checks**  
   Ensure `forward` ran by verifying cached tensors.
2. **Activation derivative**  
   `activation_prime = activation.derivative(self.last_output)` returns a mask or slope per neuron.
3. **Delta computation**  
   `delta = upstream_gradient * activation_prime` combines the incoming gradient with the activation slope.
4. **Gradients for parameters**  
   - `grad_weights = (last_input.T @ delta) / batch_size`  
   - `grad_biases = sum(delta, axis=0, keepdims=True) / batch_size`
5. **Propagate upstream**  
   Return `delta @ weights.T` so previous layers know how to adjust.

### Tips for Experiments

- Swap activations (ReLU, sigmoid) to see how the layer behaves.
- Adjust width (number of units) to test model capacity.
- Inspect `grad_weights` to ensure gradients are non-zero; otherwise check activation saturation.

### Linking Forward

The final layer converts hidden representations into predictions. Continue with `output-layer.md` to study the readout stage.
