from enum import Enum

from tinygrad import Tensor

from tinyops.ops.image._filtering import apply_morphological_filter


class MorphologicalOperation(Enum):
    """Types of morphological operations."""
    OPEN = "open"
    CLOSE = "close"
    GRADIENT = "gradient"
    TOP_HAT = "top_hat"
    BLACK_HAT = "black_hat"


def morphological_erode(image: Tensor, structuring_element: Tensor) -> Tensor:
    """Erode an image using a structuring element (local minimum).

    Args:
        image: Input image tensor (H, W) or (H, W, C).
        structuring_element: 2D kernel tensor.

    Returns:
        Eroded image tensor.
    """
    return apply_morphological_filter(image, structuring_element, "min")


def morphological_dilate(image: Tensor, structuring_element: Tensor) -> Tensor:
    """Dilate an image using a structuring element (local maximum).

    Args:
        image: Input image tensor (H, W) or (H, W, C).
        structuring_element: 2D kernel tensor.

    Returns:
        Dilated image tensor.
    """
    return apply_morphological_filter(image, structuring_element, "max")


def morphological_operation(
    image: Tensor,
    operation: MorphologicalOperation,
    structuring_element: Tensor,
) -> Tensor:
    """Perform a compound morphological transformation.

    Args:
        image: Input image tensor.
        operation: Type of morphological operation.
        structuring_element: 2D kernel tensor.

    Returns:
        Processed image tensor.
    """
    if operation == MorphologicalOperation.OPEN:
        return morphological_dilate(morphological_erode(image, structuring_element), structuring_element)
    elif operation == MorphologicalOperation.CLOSE:
        return morphological_erode(morphological_dilate(image, structuring_element), structuring_element)
    elif operation == MorphologicalOperation.GRADIENT:
        return morphological_dilate(image, structuring_element) - morphological_erode(image, structuring_element)
    elif operation == MorphologicalOperation.TOP_HAT:
        return image - morphological_dilate(morphological_erode(image, structuring_element), structuring_element)
    elif operation == MorphologicalOperation.BLACK_HAT:
        return morphological_erode(morphological_dilate(image, structuring_element), structuring_element) - image
    else:
        raise ValueError(f"Invalid morphological operation: {operation}")
