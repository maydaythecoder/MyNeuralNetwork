# Optimizer (Stochastic Gradient Descent)

## Why It Matters

An optimizer turns loss gradients into parameter updates. Without it, the network would know it is wrong but never learn. This project implements Stochastic Gradient Descent (SGD) with optional gradient clipping.

### Components Covered

- `Optimizer` protocol: requires a `step(params_and_grads)` method.
- `SGD` dataclass: keeps hyperparameters (`learning_rate`, `max_grad_norm`) and applies updates.

### Step-by-Step Walkthrough

1. **Protocol contract**  
   The `Optimizer` protocol in `logic/Optimizer.py` ensures any optimizer exposes a `step` method that mutates parameters in-place.
2. **Initializer checks**  
   `SGD.__post_init__` validates that learning rate and clipping threshold (if provided) are positive. Catching misconfigurations early avoids silent failures.
3. **Update loop**  
   `step` receives an iterable of `(parameters, gradients)` NumPy arrays. For each pair:  
   - Optionally clip gradients to keep their magnitude below `max_grad_norm`. This prevents exploding gradients.  
   - Apply `parameters -= learning_rate * gradients`.
4. **Gradient clipping helper**  
   `_clip_gradients` computes the Euclidean norm. If the norm exceeds the threshold, it scales the gradient vector to fit the limit.

### How It Connects

The optimizer consumes the gradients produced by the loss and layers. The container that bundles parameters and their gradients is the `Node` class—continue with `parameter-node.md` to see how parameters are stored.
