"""Internal helpers for image filtering operations."""

from tinygrad import Tensor, dtypes

from tinyops.ops.image.pad import PaddingMode, pad_image


def _same_size_padding(kernel_height: int, kernel_width: int) -> tuple[int, int, int, int]:
    """Padding (left, top, right, bottom) that keeps spatial size for a 2D kernel."""
    pad_top = kernel_height // 2
    pad_bottom = (kernel_height - 1) // 2
    pad_left = kernel_width // 2
    pad_right = (kernel_width - 1) // 2
    return (pad_left, pad_top, pad_right, pad_bottom)


def _pad_for_convolution(
    image: Tensor,
    padding: tuple[int, int, int, int],
    border_mode: PaddingMode,
) -> tuple[Tensor, tuple[int, int, int, int]]:
    """Apply border padding; return (padded_image, conv2d_padding).

    ``PaddingMode.CONSTANT`` leaves spatial padding to ``conv2d`` (zero fill).
    Other modes use :func:`pad_image` and pass zero ``conv2d`` padding.
    """
    if border_mode == PaddingMode.CONSTANT:
        # conv2d padding is (left, right, top, bottom)
        conv_padding = (padding[0], padding[2], padding[1], padding[3])
        return image, conv_padding

    if image.ndim == 4:
        permuted = image.permute(1, 2, 0, 3)
        padded_permuted = pad_image(permuted, padding, padding_mode=border_mode)
        padded = padded_permuted.permute(2, 0, 1, 3)
    else:
        padded = pad_image(image, padding, padding_mode=border_mode)
    return padded, (0, 0, 0, 0)


def _reshape_for_grouped_convolution(
    padded: Tensor,
) -> tuple[Tensor, int, tuple[int, ...] | None]:
    """Reshape image to ``(N, C, H, W)`` for depthwise grouped conv.

    Returns ``(nchw_input, groups, restore_permutation)``. For 2D grayscale
    inputs ``restore_permutation`` is ``None`` (restore by reshape only).
    """
    if padded.ndim == 2:
        return padded.reshape(1, 1, *padded.shape), 1, None
    if padded.ndim == 3:
        return padded.permute(2, 0, 1).unsqueeze(0), padded.shape[2], (1, 2, 0)
    if padded.ndim == 4:
        return padded.permute(0, 3, 1, 2), padded.shape[3], (0, 2, 3, 1)
    raise ValueError(f"Unsupported input shape: {padded.shape}")


def _restore_from_grouped_convolution(
    output: Tensor,
    restore_permutation: tuple[int, ...] | None,
) -> Tensor:
    """Map conv ``(N, C, H, W)`` output back to the caller's layout."""
    if restore_permutation is None:
        return output.reshape(output.shape[2], output.shape[3])
    if len(restore_permutation) == 3:
        return output.squeeze(0).permute(*restore_permutation)
    return output.permute(*restore_permutation)


def apply_convolution_filter(
    image: Tensor,
    kernel: Tensor,
    scale: float = 1.0,
    delta: float = 0.0,
    border_mode: PaddingMode = PaddingMode.REFLECT,
    padding: tuple[int, int, int, int] | None = None,
) -> Tensor:
    """Apply a 2D convolution filter to an image.

    Handles grayscale (H, W), color (H, W, C) and batch (N, H, W, C) inputs
    by reshaping to (N, C, H, W) for grouped depthwise convolution.

    Args:
        image: Input image tensor.
        kernel: 2D filter kernel.
        scale: Output scale factor.
        delta: Value added to filtered results.
        border_mode: Padding mode for borders.
        padding: Explicit (left, top, right, bottom) padding, or None
            for same-size output.

    Returns:
        Filtered image tensor with the same shape as the input.
    """
    if image.dtype == dtypes.uint8:
        image = image.cast(dtypes.float32)

    kernel_height, kernel_width = kernel.shape
    if padding is None:
        padding = _same_size_padding(kernel_height, kernel_width)

    padded, conv_padding = _pad_for_convolution(image, padding, border_mode)
    input_for_conv, groups, restore_permutation = _reshape_for_grouped_convolution(padded)

    expanded_kernel = kernel.expand(groups, 1, kernel_height, kernel_width)
    output = input_for_conv.conv2d(expanded_kernel, padding=conv_padding, groups=groups)
    output = _restore_from_grouped_convolution(output, restore_permutation)

    return output * scale + delta


def apply_morphological_filter(image: Tensor, kernel: Tensor, operation: str) -> Tensor:
    """Apply a morphological filter (erosion/dilation) using sliding window.

    Args:
        image: Input image (H, W) or (H, W, C).
        kernel: Structuring element (H, W).
        operation: Either ``'min'`` (erosion) or ``'max'`` (dilation).

    Returns:
        Filtered image tensor.
    """
    kernel_height, kernel_width = kernel.shape
    pad_y = (kernel_height - 1) // 2
    pad_x = (kernel_width - 1) // 2

    original_height, original_width = image.shape[:2]

    if image.ndim == 2:
        pad_config = ((pad_y, pad_y), (pad_x, pad_x))
    elif image.ndim == 3:
        pad_config = ((pad_y, pad_y), (pad_x, pad_x), (0, 0))
    else:
        raise NotImplementedError(f"Morphological filter not implemented for ndim={image.ndim}")

    fill_value = float("inf") if operation == "min" else float("-inf")
    padded = image.pad(pad_config, value=fill_value)

    views = []
    for row in range(kernel_height):
        for column in range(kernel_width):
            if image.ndim == 2:
                view = padded[row : row + original_height, column : column + original_width]
            else:
                view = padded[row : row + original_height, column : column + original_width, :]
            views.append(view.unsqueeze(0))

    stacked = Tensor.cat(*views, dim=0)
    mask_shape = (kernel_height * kernel_width,) + (1,) * image.ndim
    kernel_mask = kernel.flatten().reshape(mask_shape) > 0
    masked = Tensor.where(kernel_mask, stacked, fill_value)

    if operation == "min":
        return masked.min(axis=0)
    elif operation == "max":
        return masked.max(axis=0)
    else:
        raise ValueError(f"Invalid operation: {operation}")
