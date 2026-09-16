# tinyops

Complex operations implemented purely in `tinygrad`.

## Methodology

This is an experimental project. The goal is to reconstruct algorithms from domains such as computer vision, audio, and machine learning using strictly `tinygrad` primitives.

By avoiding external runtime libraries, we ensure that the entire execution flow passes through the `tinygrad` graph. This allows for aggressive **kernel fusion**, where multiple operations are compiled into a single GPU/accelerator kernel, maximizing efficiency and portability.

## Dependencies

The only runtime dependency is **tinygrad**.

Libraries such as `numpy`, `opencv-python`, and `scikit-learn` are development dependencies, used exclusively in tests to verify the numerical and functional parity of the implementations.

## Development

We use `mise` to manage the development environment.

```bash
# Install dependencies
mise install

# Run tests
mise run test
```
