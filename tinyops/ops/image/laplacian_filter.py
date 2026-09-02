from tinygrad import Tensor, dtypes

from tinyops.ops.image._filtering import apply_convolution_filter

_LAPLACIAN_KERNEL = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]


def laplacian_filter(
    image: Tensor,
    kernel_size: int = 1,
    scale: float = 1.0,
    delta: float = 0.0,
) -> Tensor:
    """Compute the Laplacian of an image.

    Args:
        image: Input image tensor.
        kernel_size: Aperture size (must be 1, uses 3x3 kernel).
        scale: Scale factor for output.
        delta: Value added to output.

    Returns:
        Laplacian image tensor.

    Raises:
        ValueError: If kernel_size is not 1.
    """
    if kernel_size != 1:
        raise NotImplementedError(f"Laplacian kernel for kernel_size={kernel_size} is not implemented.")

    input_dtype = image.dtype
    compute_dtype = dtypes.float32 if input_dtype == dtypes.uint8 else input_dtype
    kernel = Tensor(_LAPLACIAN_KERNEL, dtype=compute_dtype)
    return apply_convolution_filter(image, kernel, scale, delta)
