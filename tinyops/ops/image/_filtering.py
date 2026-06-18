"""Internal helpers for image filtering operations."""

from tinygrad import Tensor, dtypes

from tinyops.ops.image.pad import pad_image, PaddingMode


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
    input_dtype = image.dtype
    if input_dtype == dtypes.uint8:
        image = image.cast(dtypes.float32)

    kernel_height, kernel_width = kernel.shape

    if padding is None:
        pad_top = kernel_height // 2
        pad_bottom = (kernel_height - 1) // 2
        pad_left = kernel_width // 2
        pad_right = (kernel_width - 1) // 2
        padding = (pad_left, pad_top, pad_right, pad_bottom)

    conv_padding = (0, 0, 0, 0)
    padded = image

    if border_mode == PaddingMode.CONSTANT:
        conv_padding = (padding[0], padding[2], padding[1], padding[3])
    else:
        if image.ndim == 4:
            permuted = image.permute(1, 2, 0, 3)
            padded_permuted = pad_image(permuted, padding, padding_mode=border_mode)
            padded = padded_permuted.permute(2, 0, 1, 3)
        else:
            padded = pad_image(image, padding, padding_mode=border_mode)

    if padded.ndim == 2:
        input_for_conv = padded.reshape(1, 1, *padded.shape)
        groups = 1
        original_permutation = None
    elif padded.ndim == 3:
        input_for_conv = padded.permute(2, 0, 1).unsqueeze(0)
        groups = padded.shape[2]
        original_permutation = (1, 2, 0)
    elif padded.ndim == 4:
        input_for_conv = padded.permute(0, 3, 1, 2)
        groups = padded.shape[3]
        original_permutation = (0, 2, 3, 1)
    else:
        raise ValueError(f"Unsupported input shape: {image.shape}")

    expanded_kernel = kernel.expand(groups, 1, kernel_height, kernel_width)
    output = input_for_conv.conv2d(expanded_kernel, padding=conv_padding, groups=groups)

    if original_permutation:
        if len(original_permutation) == 3:
            output = output.squeeze(0).permute(*original_permutation)
        else:
            output = output.permute(*original_permutation)
    else:
        output = output.reshape(output.shape[2], output.shape[3])

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
