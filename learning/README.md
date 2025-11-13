# Learning Roadmap

Welcome to the guided tour of the dense neural network project. These notes assume you are comfortable with basic Python (functions, lists, NumPy arrays) and want a structured path from the smallest building blocks up to the full training loop.

- Start at `activation-functions.md` to see how individual math functions shape neuron outputs.
- Move on to `loss-functions.md` to learn how the network measures its mistakes.
- Continue with `optimizer.md` to understand how mistakes push parameters in better directions.
- Read `parameter-node.md` to see how weights and biases live together with their gradients.
- Study the three layer docs in order: `input-layer.md`, `hidden-layer.md`, and `output-layer.md`.
- Finish with `network-assembly.md` for the big picture: training flow, gradient checks, and demo tools.

Each document ends with “Linking Forward” so you always know why the next topic matters.

## Suggested Study Flow

1. `activation-functions.md`
2. `loss-functions.md`
3. `optimizer.md`
4. `parameter-node.md`
5. `input-layer.md`
6. `hidden-layer.md`
7. `output-layer.md`
8. `network-assembly.md`

### What You Will Achieve

- Understand every moving part used in a dense neural network.
- Learn how forward and backward passes work without relying on a black-box framework.
- Gain confidence to experiment with new architectures or activation choices.

### How to Study

- Keep the corresponding source file open while you read.
- Reproduce the formulas in a notebook or REPL to see the numbers change.
- Try modifying hyperparameters (learning rate, hidden units) once you reach the full network section.
