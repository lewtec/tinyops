from typing import Tuple, Union

from tinygrad import Tensor


def center_crop(x: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    """Crops the given image at the center.

    Args:
        x: Image to be cropped. Shape is (C, H, W).
        output_size: Desired output size of the crop. If int, a square crop is made.

    Returns:
        Cropped image.
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    h, w = x.shape[-2:]
    th, tw = output_size

    if h < th or w < tw:
        raise ValueError(f"Target size ({th}, {tw}) is larger than input size ({h}, {w})")

    i = (h - th) // 2
    j = (w - tw) // 2

    return x[..., i : i + th, j : j + tw]
