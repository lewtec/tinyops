from enum import Enum

from tinygrad import Tensor


class ConvolutionMode(Enum):
    """Output size mode for 1D convolution."""

    FULL = "full"
    VALID = "valid"
    SAME = "same"


def convolution_1d(
    signal: Tensor,
    kernel: Tensor,
    mode: ConvolutionMode = ConvolutionMode.FULL,
) -> Tensor:
    """Compute the discrete linear convolution of two 1D sequences.

    Args:
        signal: First 1D input tensor.
        kernel: Second 1D input tensor.
        mode: Output size mode (FULL, VALID, or SAME).

    Returns:
        Convolution result tensor.

    Raises:
        ValueError: If inputs are not 1D or are empty.
    """
    if signal.ndim != 1 or kernel.ndim != 1:
        raise ValueError("Input tensors must be one-dimensional.")

    signal_length = signal.shape[0]
    kernel_length = kernel.shape[0]

    if signal_length == 0:
        raise ValueError("signal cannot be empty")
    if kernel_length == 0:
        raise ValueError("kernel cannot be empty")

    # Ensure signal is the longer tensor
    if kernel_length > signal_length:
        signal, kernel = kernel, signal
        signal_length, kernel_length = kernel_length, signal_length

    if mode == ConvolutionMode.FULL:
        pad_left = kernel_length - 1
        pad_right = kernel_length - 1
    elif mode == ConvolutionMode.SAME:
        pad_left = kernel_length // 2
        pad_right = (kernel_length - 1) // 2
    elif mode == ConvolutionMode.VALID:
        pad_left = 0
        pad_right = 0
    else:
        raise ValueError(f"Invalid mode '{mode}'")

    x = signal.reshape(1, 1, 1, -1)
    w = kernel.flip(0).reshape(1, 1, 1, -1)

    out = x.conv2d(w, padding=(pad_left, pad_right, 0, 0))
    return out.flatten()
