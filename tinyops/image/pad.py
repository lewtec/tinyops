from typing import Sequence, Union

from tinygrad import Tensor


def pad(
    x: Tensor,
    padding: Union[int, Sequence[int]],
    fill: float = 0,
    padding_mode: str = "constant",
) -> Tensor:
    """Pad the given image on all sides with specified padding mode.

    Args:
        x: Image to be padded. Shape is (..., H, W).
        padding: Padding on each border. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is
            provided this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Value for the padded pixels.
        padding_mode: Type of padding. Should be: "constant", "reflect", "replicate" or "circular".

    Returns:
        Padded image.
    """
    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if isinstance(padding, (tuple, list)) and len(padding) not in [1, 2, 4]:
        raise ValueError(
            f"Padding must be an int or a 1, 2, or 4 item sequence, but got {len(padding)}"
        )

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif isinstance(padding, (list, tuple)):
        if len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        else:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

    if padding_mode == "replicate":
        # Vertical padding
        parts_v = []
        if pad_top > 0:
            slicer = [slice(None)] * x.ndim
            slicer[-2] = slice(0, 1)
            repeats = [1] * x.ndim
            repeats[-2] = pad_top
            parts_v.append(x[tuple(slicer)].repeat(repeats))

        parts_v.append(x)

        if pad_bottom > 0:
            slicer = [slice(None)] * x.ndim
            slicer[-2] = slice(x.shape[-2] - 1, x.shape[-2])
            repeats = [1] * x.ndim
            repeats[-2] = pad_bottom
            parts_v.append(x[tuple(slicer)].repeat(repeats))

        y = Tensor.cat(*parts_v, dim=-2) if len(parts_v) > 1 else x

        # Horizontal padding
        parts_h = []
        if pad_left > 0:
            slicer = [slice(None)] * y.ndim
            slicer[-1] = slice(0, 1)
            repeats = [1] * y.ndim
            repeats[-1] = pad_left
            parts_h.append(y[tuple(slicer)].repeat(repeats))

        parts_h.append(y)

        if pad_right > 0:
            slicer = [slice(None)] * y.ndim
            slicer[-1] = slice(y.shape[-1] - 1, y.shape[-1])
            repeats = [1] * y.ndim
            repeats[-1] = pad_right
            parts_h.append(y[tuple(slicer)].repeat(repeats))

        return Tensor.cat(*parts_h, dim=-1) if len(parts_h) > 1 else y

    paddings = ((0, 0),) * (x.ndim - 2) + (
        (pad_top, pad_bottom),
        (pad_left, pad_right),
    )

    if padding_mode == "constant":
        return x.pad(paddings, value=fill)
    elif padding_mode == "reflect":
        return x.pad(paddings, mode="reflect")
    elif padding_mode == "circular":
        # Horizontal padding
        parts_h = []
        if pad_left > 0:
            parts_h.append(x[..., -pad_left:])
        parts_h.append(x)
        if pad_right > 0:
            parts_h.append(x[..., :pad_right])
        y = Tensor.cat(*parts_h, dim=-1) if len(parts_h) > 1 else x

        # Vertical padding
        parts_v = []
        if pad_top > 0:
            parts_v.append(y[..., -pad_top:, :])
        parts_v.append(y)
        if pad_bottom > 0:
            parts_v.append(y[..., :pad_bottom, :])

        return Tensor.cat(*parts_v, dim=-2) if len(parts_v) > 1 else y

    else:
        raise ValueError(
            f"Padding mode '{padding_mode}' is not supported. Choose from 'constant', 'reflect', 'replicate', 'circular'."
        )
