from tinygrad import Tensor, dtypes
from tinyops.linalg import inv

def warp_affine(src: Tensor, M: Tensor, dsize: tuple[int, int], border_value: float = 0.0) -> Tensor:
    """
    Applies an affine transformation to an image.

    Args:
        src (Tensor): The input image. Shape: (H, W, C) or (H, W).
        M (Tensor): The 2x3 transformation matrix.
        dsize (tuple[int, int]): The size of the output image (W, H).
        border_value (float): The value to use for pixels outside the source image.

    Returns:
        Tensor: The transformed image.
    """
    H, W = src.shape[:2]
    C = src.shape[2] if len(src.shape) == 3 else 1
    out_W, out_H = dsize

    # Augment M to a 3x3 matrix for inversion
    M_aug = Tensor.cat(M.cast(dtypes.float32), Tensor([[0, 0, 1]]), dim=0)

    # Invert the matrix
    M_inv = inv(M_aug)
    M_transform = M_inv[0:2, 0:3]

    # Pad the source image
    if len(src.shape) == 3:
        padding = ((1, 1), (1, 1), (0, 0))
    else:
        padding = ((1, 1), (1, 1))
    padded_src = src.pad(padding, value=border_value)
    Hp, Wp = padded_src.shape[:2]

    # Create a grid of coordinates for the output image
    y, x = Tensor.arange(out_H).cast(dtypes.float32), Tensor.arange(out_W).cast(dtypes.float32)
    gy, gx = y.reshape(-1, 1).expand(out_H, out_W), x.reshape(1, -1).expand(out_H, out_W)

    # Add a row of ones to the coordinates to make them homogeneous
    coords = Tensor.stack([gx.flatten(), gy.flatten(), Tensor.ones(out_H * out_W)]).cast(dtypes.float32)

    # Transform the coordinates to the original source image space
    transformed_coords = M_transform.matmul(coords)
    src_x_orig, src_y_orig = transformed_coords[0], transformed_coords[1]

    # Shift coordinates to the padded image space
    src_x = src_x_orig + 1
    src_y = src_y_orig + 1

    # Bilinear interpolation
    x0 = src_x.floor().cast(dtypes.int32)
    y0 = src_y.floor().cast(dtypes.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clip coordinates to be within the padded image boundaries
    x0c = x0.clip(0, Wp - 1)
    y0c = y0.clip(0, Hp - 1)
    x1c = x1.clip(0, Wp - 1)
    y1c = y1.clip(0, Hp - 1)

    # Flatten the padded image for gathering
    flat_padded_src = padded_src.reshape(Hp * Wp, C)

    # Calculate indices for gathering
    idx00 = y0c * Wp + x0c
    idx01 = y0c * Wp + x1c
    idx10 = y1c * Wp + x0c
    idx11 = y1c * Wp + x1c

    # Expand indices for gathering all channels
    gather_idx00 = idx00.reshape(-1, 1).expand(-1, C).contiguous()
    gather_idx01 = idx01.reshape(-1, 1).expand(-1, C).contiguous()
    gather_idx10 = idx10.reshape(-1, 1).expand(-1, C).contiguous()
    gather_idx11 = idx11.reshape(-1, 1).expand(-1, C).contiguous()

    # Gather pixel values
    p00 = flat_padded_src.gather(0, gather_idx00).reshape(out_H, out_W, C)
    p01 = flat_padded_src.gather(0, gather_idx01).reshape(out_H, out_W, C)
    p10 = flat_padded_src.gather(0, gather_idx10).reshape(out_H, out_W, C)
    p11 = flat_padded_src.gather(0, gather_idx11).reshape(out_H, out_W, C)

    # Calculate interpolation weights
    w00 = ((x1.cast(dtypes.float32) - src_x) * (y1.cast(dtypes.float32) - src_y)).reshape(out_H, out_W, 1).cast(dtypes.float32)
    w01 = ((src_x - x0.cast(dtypes.float32)) * (y1.cast(dtypes.float32) - src_y)).reshape(out_H, out_W, 1).cast(dtypes.float32)
    w10 = ((x1.cast(dtypes.float32) - src_x) * (src_y - y0.cast(dtypes.float32))).reshape(out_H, out_W, 1).cast(dtypes.float32)
    w11 = ((src_x - x0.cast(dtypes.float32)) * (src_y - y0.cast(dtypes.float32))).reshape(out_H, out_W, 1).cast(dtypes.float32)

    # Perform the interpolation
    output_image = w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11

    if len(src.shape) == 2:
        output_image = output_image.reshape(out_H, out_W)

    return output_image
