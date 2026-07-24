"""Shared helpers for discrete Fourier transform paths."""

from collections.abc import Callable

from tinygrad import Tensor


def complex_conjugate(complex_values: Tensor) -> Tensor:
    """Negate the imaginary part of packed ``(real, imag)`` complex values.

    Args:
        complex_values: Tensor whose last axis has length 2 and holds
            ``(real, imaginary)`` components.

    Returns:
        Tensor of the same shape with imaginary components negated.
    """
    return Tensor.stack([complex_values[..., 0], -complex_values[..., 1]], dim=-1)


def separable_two_dimensional_transform(
    complex_grid: Tensor,
    one_dimensional_transform: Callable[[Tensor], Tensor],
) -> Tensor:
    """Apply a 1D complex transform separably along width, then height.

    Shared path for 2D DFT and inverse DFT. Each row (then each column of
    the intermediate) is a length-``N`` spectrum packed as ``(N, 2)``.

    Args:
        complex_grid: Input of shape ``(H, W, 2)`` with ``(real, imag)`` pairs.
        one_dimensional_transform: Callable applied to each 1D complex vector.

    Returns:
        Transformed grid of shape ``(H, W, 2)``.

    Raises:
        ValueError: If *complex_grid* is not shaped ``(H, W, 2)``.
    """
    if complex_grid.ndim != 3 or complex_grid.shape[-1] != 2:
        raise ValueError(f"input must have shape (H, W, 2), got {complex_grid.shape}")

    height, width, _ = complex_grid.shape
    if height == 0 or width == 0:
        return complex_grid

    after_width = Tensor.stack(
        [one_dimensional_transform(complex_grid[row_index]) for row_index in range(height)],
        dim=0,
    )
    return Tensor.stack(
        [
            one_dimensional_transform(after_width[:, column_index, :])
            for column_index in range(width)
        ],
        dim=1,
    )
