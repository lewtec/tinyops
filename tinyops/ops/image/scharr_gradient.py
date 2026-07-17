from tinygrad import Tensor

from tinyops.ops.image.sobel_gradient import (
    GradientDirection,
    _apply_directional_gradient,
)

_SCHARR_HORIZONTAL = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
_SCHARR_VERTICAL = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]


def scharr_gradient(
    image: Tensor,
    direction: GradientDirection,
    scale: float = 1.0,
    delta: float = 0.0,
) -> Tensor:
    """Compute image gradient using the Scharr operator.

    The Scharr operator provides better rotational symmetry than Sobel.

    Args:
        image: Input image tensor.
        direction: Gradient direction (HORIZONTAL or VERTICAL).
        scale: Scale factor for output.
        delta: Value added to output.

    Returns:
        Gradient image tensor.
    """
    return _apply_directional_gradient(
        image,
        direction,
        _SCHARR_HORIZONTAL,
        _SCHARR_VERTICAL,
        scale=scale,
        delta=delta,
    )
