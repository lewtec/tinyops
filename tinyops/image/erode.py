from tinygrad import Tensor

from tinyops.image._utils import apply_morphological_filter


def erode(x: Tensor, kernel: Tensor) -> Tensor:
    """
    Erodes an image using a specific structuring element.
    Computes the local minimum over the area of the kernel.

    Args:
        x: Input image tensor.
           Currently supports shapes (H, W) or (H, W, C).
        kernel: Structuring element (2D tensor).
           The shape of the kernel determines the neighborhood size.
           Non-zero values in the kernel indicate the neighborhood to consider.

    Returns:
        Eroded image tensor with the same shape as `x`.
        Padding is handled by filling borders with infinity, so minimum is computed only on valid pixels.

    Raises:
        NotImplementedError: If `x` has unsupported dimensions (e.g., 4D).
    """
    return apply_morphological_filter(x, kernel, mode="min")
