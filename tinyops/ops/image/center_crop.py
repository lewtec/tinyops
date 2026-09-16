from tinygrad import Tensor


def center_crop(image: Tensor, output_size: int | tuple[int, int]) -> Tensor:
    """Crop the center of an image to the given size.

    If the image is smaller than the requested size, it is padded with
    zeros first.

    Args:
        image: Image tensor of shape (..., H, W).
        output_size: Desired crop size. An int gives a square crop.

    Returns:
        Cropped image tensor.
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    height, width = image.shape[-2:]
    target_height, target_width = output_size

    if height < target_height or width < target_width:
        pad_top = max(0, (target_height - height) // 2)
        pad_bottom = max(0, target_height - height - pad_top)
        pad_left = max(0, (target_width - width) // 2)
        pad_right = max(0, target_width - width - pad_left)

        padding_config = tuple([(0, 0)] * (image.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)])
        image = image.pad(padding_config)
        height, width = image.shape[-2:]

    row_start = (height - target_height) // 2
    column_start = (width - target_width) // 2

    return image[..., row_start : row_start + target_height, column_start : column_start + target_width]
