# Parameter Node

## Why It Matters

Layers need somewhere to store weights, biases, and their gradients. The `Node` class is that container. Keeping parameters together simplifies optimizer code and makes gradient bookkeeping explicit.

### Components Covered

- `Node` dataclass in `structure/Node.py`.
- He initialization strategy via `Node.initialize`.
- Gradient buffers and helper utilities.

### Step-by-Step Walkthrough

1. **Structure**  
   A `Node` holds four NumPy arrays: `weights`, `biases`, `grad_weights`, and `grad_biases`. The last two are created in `__post_init__` and start filled with zeros.
2. **Initialization**  
   `Node.initialize(input_dim, output_dim, rng)` generates weights from a normal distribution with standard deviation `sqrt(2 / input_dim)` (He initialization). Biases start at zero. This stabilizes activations when using ReLU.
3. **Gradient reset**  
   `zero_gradients` fills the gradient arrays with zeros. Layers call this after each optimizer step to avoid stale gradients.
4. **Shape inspection**  
   `parameter_shapes` is a debugging helper that returns the shapes of weights and biases, useful when adding new layers.

### Linking Forward

With parameters in place, it is time to see the simplest layer that uses them: the `InputLayer`. Continue with `input-layer.md`.
