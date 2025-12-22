from tinygrad import Tensor

def erode(x: Tensor, kernel: Tensor) -> Tensor:
    """
    Erodes an image by using a specific structuring element.
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb
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
        raise NotImplementedError(f"erode not implemented for ndim={x.ndim}")

    x_padded = x.pad(padding, value=float('inf'))

    views = []
    for i in range(h_k):
        for j in range(w_k):
            if x.ndim == 2:
                view = x_padded[i:i + h_orig, j:j + w_orig]
            else:
                view = x_padded[i:i + h_orig, j:j + w_orig, :]
            views.append(view.unsqueeze(0))

    stacked_views = Tensor.cat(*views, dim=0)

    mask_shape = (h_k * w_k,) + (1,) * x.ndim
    kernel_mask = kernel.flatten().reshape(mask_shape) > 0

    masked_views = Tensor.where(kernel_mask, stacked_views, float('inf'))

    eroded = masked_views.min(axis=0)

    return eroded
