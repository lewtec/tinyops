from tinygrad import Tensor


def center_crop(x: Tensor, size: int | tuple[int, int]) -> Tensor:
    """
    Crops the given image tensor at the center.
    Args:
      x: Image tensor of shape (..., H, W).
      size: Desired output size of the crop. If size is an int, a square crop (size, size) is made.
    """
    if isinstance(size, int):
        size = (size, size)

    h, w = x.shape[-2:]
    th, tw = size

    if h < th or w < tw:
        pad_h_top = max(0, (th - h) // 2)
        pad_h_bottom = max(0, th - h - pad_h_top)
        pad_w_left = max(0, (tw - w) // 2)
        pad_w_right = max(0, tw - w - pad_w_left)

        padding_config = tuple([(0, 0)] * (x.ndim - 2) + [(pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)])
        x = x.pad(padding_config)
        h, w = x.shape[-2:]

    i = (h - th) // 2
    j = (w - tw) // 2

    return x[..., i : i + th, j : j + tw]
