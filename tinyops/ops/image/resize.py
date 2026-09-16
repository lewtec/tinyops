from enum import Enum

from tinygrad import Tensor, dtypes


class InterpolationMethod(Enum):
    """Supported interpolation methods for image resizing."""

    NEAREST_NEIGHBOR = "nearest_neighbor"
    BILINEAR = "bilinear"


def _resize_nearest(
    image: Tensor,
    output_size: tuple[int, int],
    input_size: tuple[int, int],
    coords: tuple[Tensor, Tensor],
) -> Tensor:
    output_height, output_width = output_size
    input_height, input_width = input_size
    row_coords, column_coords = coords

    scale_y = input_height / output_height
    scale_x = input_width / output_width
    source_y = (row_coords.cast(dtypes.float32) * scale_y).floor().cast(dtypes.int32).clip(0, input_height - 1)
    source_x = (column_coords.cast(dtypes.float32) * scale_x).floor().cast(dtypes.int32).clip(0, input_width - 1)
    return image[source_y, source_x]


def _resize_bilinear(
    image: Tensor,
    output_size: tuple[int, int],
    input_size: tuple[int, int],
    coords: tuple[Tensor, Tensor],
) -> Tensor:
    output_height, output_width = output_size
    input_height, input_width = input_size
    row_coords, column_coords = coords

    scale_y = input_height / output_height
    scale_x = input_width / output_width
    source_y = ((row_coords.cast(dtypes.float32) + 0.5) * scale_y - 0.5).clip(0, input_height - 1)
    source_x = ((column_coords.cast(dtypes.float32) + 0.5) * scale_x - 0.5).clip(0, input_width - 1)

    y_floor = source_y.floor().cast(dtypes.int32)
    x_floor = source_x.floor().cast(dtypes.int32)
    dy = (source_y - y_floor).unsqueeze(2)
    dx = (source_x - x_floor).unsqueeze(2)

    y_ceil = (y_floor + 1).clip(0, input_height - 1)
    x_ceil = (x_floor + 1).clip(0, input_width - 1)

    top_left = image[y_floor, x_floor]
    top_right = image[y_floor, x_ceil]
    bottom_left = image[y_ceil, x_floor]
    bottom_right = image[y_ceil, x_ceil]

    return (
        top_left * (1 - dy) * (1 - dx)
        + top_right * (1 - dy) * dx
        + bottom_left * dy * (1 - dx)
        + bottom_right * dy * dx
    )


def resize_image(
    image: Tensor,
    target_size: tuple[int, int],
    method: InterpolationMethod = InterpolationMethod.BILINEAR,
) -> Tensor:
    """Resize an image to a target size.

    Args:
        image: Input image tensor (H, W) or (H, W, C).
        target_size: Desired output size as (height, width).
        method: Interpolation method.

    Returns:
        Resized image tensor.
    """
    is_grayscale = image.ndim == 2
    if is_grayscale:
        image = image.unsqueeze(2)

    input_height, input_width, _ = image.shape
    output_height, output_width = target_size

    row_coords, column_coords = Tensor.meshgrid(
        Tensor.arange(output_height), Tensor.arange(output_width), indexing="ij"
    )

    if method == InterpolationMethod.NEAREST_NEIGHBOR:
        result = _resize_nearest(
            image, (output_height, output_width), (input_height, input_width), (row_coords, column_coords)
        )
    elif method == InterpolationMethod.BILINEAR:
        result = _resize_bilinear(
            image, (output_height, output_width), (input_height, input_width), (row_coords, column_coords)
        )
    else:
        raise NotImplementedError(f"Interpolation method {method} is not supported.")

    if is_grayscale:
        result = result.squeeze(2)

    return result
