from tinygrad import Tensor


def hamming_distance(first: Tensor, second: Tensor) -> Tensor:
    """Compute the Hamming distance between two 1D tensors.

    The Hamming distance is the proportion of positions where the
    elements differ.

    Args:
        first: First 1D tensor.
        second: Second 1D tensor.

    Returns:
        Scalar tensor with the Hamming distance.

    Raises:
        ValueError: If inputs are not 1D or have different lengths.
    """
    if len(first.shape) != 1 or len(second.shape) != 1:
        raise ValueError("Input tensors must be 1-D.")
    if first.shape[0] != second.shape[0]:
        raise ValueError("Input tensors must have the same length.")

    return (first != second).float().mean()
