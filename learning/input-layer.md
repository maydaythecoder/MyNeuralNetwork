# Input Layer

## Role in the Network

The `InputLayer` is the gateway that checks whether the incoming data matches the expected feature dimension. It does not change the data; it simply caches the batch for later use.

### Forward Walkthrough

1. Receives a NumPy array of shape `(batch_size, input_dim)`.
2. Validates that `inputs.shape[1]` equals the configured `input_dim`.
3. Stores the inputs in `last_output` for debugging or downstream logging.
4. Returns the inputs unchanged.

### Backward Walkthrough

1. Accepts `upstream_gradient`, which has the same shape as the inputs.
2. Since the input layer has no parameters, it forwards the gradient without modification.

### Common Use Cases

- Acts as a guardrail when experimenting with new datasets or feature engineering.
- Provides a consistent interface so the rest of the network always receives validated data.

### Linking Forward

Once the data passes validation, it flows into learnable layers. Start with `hidden-layer.md` to understand how the first trainable block works.
