# Loss Functions

## Why They Matter

Loss functions measure how wrong the network’s predictions are. The optimizer uses this single number to decide how to adjust the weights. If activations shape signals, losses judge them.

### Components Covered

- `Loss` protocol: enforces `__call__(predictions, targets)` and `derivative(...)`.
- `LossFunction` dataclass: mirrors `UnaryActivation`, keeping forward and backward logic together.
- Implementations: `mean_squared_error`, `cross_entropy`.

### Step-by-Step Walkthrough

1. **Interface**  
   `Loss` lives in `logic/lossFunction.py`. Every loss must return a scalar score and provide a gradient with respect to the predictions.
2. **Wrapper**  
   `LossFunction` stores two callables. The forward returns a Python float; the backward returns a NumPy array matching the predictions’ shape.
3. **Mean Squared Error (MSE)**  
   - Forward: subtract targets from predictions, square the residuals, average them.  
   - Backward: `2 / batch_size * (predictions - targets)`. Each gradient entry tells how much the prediction should move to reduce the squared error.
   - Use Cases: regression tasks where outputs are continuous numbers.
4. **Cross Entropy**  
   - Forward: clips predictions to avoid log(0), then computes `-∑ target * log(prediction)` per sample and averages.  
   - Backward: `(predictions - targets) / batch_size` (with clipping applied). Works seamlessly with softmax outputs.  
   - Use Cases: multi-class classification with one-hot targets.

### Linking Forward

Loss gradients are the signal the optimizer consumes. Move on to `optimizer.md` to learn how the gradients change the model parameters.
