from tinygrad import Tensor, dtypes

def dilate(image: Tensor, kernel: Tensor) -> Tensor:
    """
    Dilates an image by using a specific structuring element.
    This operation is equivalent to a local maximum filter.

    Args:
        image: The input image as a Tensor. Shape should be (H, W) or (C, H, W).
        kernel: The structuring element. A Tensor with shape (kH, kW).
                Non-zero values are considered part of the kernel.

    Returns:
        The dilated image as a Tensor.
    """
    if image.ndim not in [2, 3]:
        raise ValueError(f"Input image must be 2D or 3D, but got {image.ndim} dimensions.")
    if kernel.ndim != 2:
        raise ValueError(f"Kernel must be 2D, but got {kernel.ndim} dimensions.")

    is_3d = image.ndim == 3
    if is_3d:
        C, H, W = image.shape
        channels = [dilate(image[i], kernel) for i in range(C)]
        return Tensor.stack(*channels).cast(image.dtype)

    H, W = image.shape
    kH, kW = kernel.shape

    pad_value = 0 if dtypes.is_unsigned(image.dtype) else float('-inf')

    anchor_y, anchor_x = kH // 2, kW // 2

    # Pad image enough to handle all shifts. Padding by the kernel dimensions is a safe upper bound.
    padded_image = image.pad(((kH, kH), (kW, kW)), value=pad_value)

    views = []
    kernel_np = kernel.numpy()

    for r in range(kH):
        for c in range(kW):
            if kernel_np[r, c] == 0:
                continue

            # For each kernel element, calculate the required shift of the image.
            dy, dx = anchor_y - r, anchor_x - c

            # Slice the padded image to create a shifted view.
            # A positive dy shifts the image content "down", so the slice starts higher up.
            start_y, start_x = kH - dy, kW - dx
            view = padded_image[start_y:start_y + H, start_x:start_x + W]
            views.append(view)

    if not views:
        return image.copy()

    # Stack the shifted views and compute the element-wise maximum.
    stacked_views = Tensor.stack(*views)
    dilated_image = stacked_views.max(axis=0)

    return dilated_image.cast(image.dtype)
