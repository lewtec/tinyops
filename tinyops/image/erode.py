from tinygrad import Tensor, dtypes

def erode(image: Tensor, kernel: Tensor) -> Tensor:
    """
    Erodes an image by using a specific structuring element.
    This operation is equivalent to a local minimum filter.

    Args:
        image: The input image as a Tensor. Shape should be (H, W) or (C, H, W).
        kernel: The structuring element. A Tensor with shape (kH, kW).
                Non-zero values are considered part of the kernel.

    Returns:
        The eroded image as a Tensor.
    """
    if image.ndim not in [2, 3]:
        raise ValueError(f"Input image must be 2D or 3D, but got {image.ndim} dimensions.")
    if kernel.ndim != 2:
        raise ValueError(f"Kernel must be 2D, but got {kernel.ndim} dimensions.")

    is_3d = image.ndim == 3
    if is_3d:
        C, H, W = image.shape
        channels = [erode(image[i], kernel) for i in range(C)]
        return Tensor.stack(*channels).cast(image.dtype)

    H, W = image.shape
    kH, kW = kernel.shape

    pad_value = 255 if dtypes.is_unsigned(image.dtype) else float('inf')

    anchor_y, anchor_x = kH // 2, kW // 2

    # Pad image enough to handle all shifts.
    padded_image = image.pad(((kH, kH), (kW, kW)), value=pad_value)

    views = []
    kernel_np = kernel.numpy()

    for r in range(kH):
        for c in range(kW):
            if kernel_np[r, c] == 0:
                continue

            dy, dx = anchor_y - r, anchor_x - c
            start_y, start_x = kH - dy, kW - dx
            view = padded_image[start_y:start_y + H, start_x:start_x + W]
            views.append(view)

    if not views:
        return image.copy()

    # Stack the shifted views and compute the element-wise minimum.
    stacked_views = Tensor.stack(*views)
    eroded_image = stacked_views.min(axis=0)

    return eroded_image.cast(image.dtype)
