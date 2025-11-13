# Output Layer

## Why It Matters

The output layer converts the final hidden representations into predictions. In classification tasks it often uses softmax to produce probabilities that sum to one.

### Forward Pass Walkthrough

1. **Cache inputs**  
   Store the incoming activations for backpropagation.
2. **Compute logits**  
   `logits = inputs @ weights + biases`. These raw scores are not yet probabilities.
3. **Apply activation**  
   By default the layer uses `softmax`, turning logits into class probabilities. You can swap in another activation (like identity) for regression tasks.
4. **Cache outputs**  
   Save the probabilities for later use in loss and gradient calculations.

### Backward Pass Walkthrough

1. **Choose derivative strategy**  
   When paired with cross-entropy loss, set `apply_activation_derivative=False` in `backward` because the combined gradient is already simplified. Otherwise multiply by the activation derivative.
2. **Compute parameter gradients**  
   - `grad_weights = (last_input.T @ delta) / batch_size`  
   - `grad_biases = sum(delta, axis=0, keepdims=True) / batch_size`
3. **Return upstream gradient**  
   Pass `delta @ weights.T` to continue backpropagation.

### Prediction Helper

`predict` picks the index of the highest probability per sample (`argmax`) and returns it as a column vector, which makes evaluating accuracy straightforward.

### Linking Forward

You now know how data flows from inputs to predictions. The final topic ties everything together—progress to `network-assembly.md`.
