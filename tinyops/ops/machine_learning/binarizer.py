from tinygrad import Tensor


def binarizer(features: Tensor, threshold: float = 0.0) -> Tensor:
    """Binarize data by thresholding.

    Args:
        features: Input tensor.
        threshold: Values above this become 1.0, at or below become 0.0.

    Returns:
        Binarized tensor.
    """
    return Tensor.where(features > threshold, 1.0, 0.0)
