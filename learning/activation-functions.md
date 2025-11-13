# Activation Functions

## Why They Matter

Activation functions are tiny mathematical switches. They take the raw numbers produced by a layer and decide how much signal to pass forward. Without them, a neural network would just be a stack of linear equations and could only model straight lines.

### Components Covered

- `Activation` protocol: promises every activation can be called like a function and can produce its derivative.
- `UnaryActivation`: bundles `forward` and `backward` callables so layers can plug any activation in without custom code.
- Built-ins: `sigmoid`, `relu`, `softmax`.

### Step-by-Step Walkthrough

1. **Protocol contract**  
   The `Activation` protocol in `logic/activationFunction.py` defines two methods: `__call__` for forward passes and `derivative` for backpropagation. Any custom activation needs to satisfy this.
2. **Reusable wrapper**  
   `UnaryActivation` stores a `forward` and `backward` function. When a layer calls the activation, it simply invokes the wrapper. This keeps layer code small and focused.
3. **Sigmoid**  
   - Forward: clamps values to [-500, 500] to avoid overflow, then applies `1 / (1 + exp(-x))`.  
   - Backward: uses the identity `sigmoid(x) * (1 - sigmoid(x))`. Because layers store the activated output, the derivative only needs that value.
4. **ReLU (Rectified Linear Unit)**  
   - Forward: returns `max(0, x)` element-wise.  
   - Backward: marks positive activations as 1 and non-positives as 0, producing a mask.
5. **Softmax**  
   - Forward: subtracts the row-wise max for numerical stability, exponentiates, then divides by the sum. This returns probabilities that add up to 1.  
   - Backward: returns `activation * (1 - activation)` which is sufficient when softmax sits in front of cross-entropy loss.

### Linking Forward

Activations produce the signals that the loss function judges. Continue with `loss-functions.md` to see how those signals are compared to the correct answers.
