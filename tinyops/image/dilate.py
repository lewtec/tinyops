from tinygrad import Tensor


def dilate(x: Tensor, kernel: Tensor) -> Tensor:
    """
    Dilates an image using a specific structuring element.
    Computes the local maximum over the area of the kernel.

    Args:
        x: Input image tensor.
           Currently supports shapes (H, W) or (H, W, C).
        kernel: Structuring element (2D tensor).
           The shape of the kernel determines the neighborhood size.
           Non-zero values in the kernel indicate the neighborhood to consider.

    Returns:
        Dilated image tensor with the same shape as `x`.
        Padding is handled by filling borders with negative infinity, so maximum is computed only on valid pixels.

    Raises:
        NotImplementedError: If `x` has unsupported dimensions (e.g., 4D).
    """
    h_k, w_k = kernel.shape
    py, px = (h_k - 1) // 2, (w_k - 1) // 2

    if x.ndim == 2:
        padding = ((py, py), (px, px))
        h_orig, w_orig = x.shape
    elif x.ndim == 3:
        padding = ((py, py), (px, px), (0, 0))
        h_orig, w_orig, _ = x.shape
    else:
        raise NotImplementedError(f"dilate not implemented for ndim={x.ndim}")

    x_padded = x.pad(padding, value=float("-inf"))

    views = []
    for i in range(h_k):
        for j in range(w_k):
            if x.ndim == 2:
                view = x_padded[i : i + h_orig, j : j + w_orig]
            else:
                view = x_padded[i : i + h_orig, j : j + w_orig, :]
            views.append(view.unsqueeze(0))

    stacked_views = Tensor.cat(*views, dim=0)

    mask_shape = (h_k * w_k,) + (1,) * x.ndim
    kernel_mask = kernel.flatten().reshape(mask_shape) > 0

    masked_views = Tensor.where(kernel_mask, stacked_views, float("-inf"))

    dilated = masked_views.max(axis=0)

    return dilated
