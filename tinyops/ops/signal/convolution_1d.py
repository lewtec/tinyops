from enum import Enum

from tinygrad import Tensor


class ConvolutionMode(Enum):
    """Output size mode for 1D convolution."""

    FULL = "full"
    VALID = "valid"
    SAME = "same"


def _order_longer_signal_first(
    signal: Tensor,
    kernel: Tensor,
) -> tuple[Tensor, Tensor, int, int]:
    """Return (signal, kernel, signal_length, kernel_length) with signal longer.

    Discrete linear convolution is commutative; swapping keeps the longer
    sequence as the conv input so padding math stays kernel-relative.
    """
    signal_length = signal.shape[0]
    kernel_length = kernel.shape[0]
    if kernel_length > signal_length:
        return kernel, signal, kernel_length, signal_length
    return signal, kernel, signal_length, kernel_length


def _convolution_1d_padding(
    mode: ConvolutionMode,
    kernel_length: int,
) -> tuple[int, int]:
    """Left/right spatial padding for the 1D signal given *mode* and kernel size."""
    if mode == ConvolutionMode.FULL:
        return kernel_length - 1, kernel_length - 1
    if mode == ConvolutionMode.SAME:
        return kernel_length // 2, (kernel_length - 1) // 2
    if mode == ConvolutionMode.VALID:
        return 0, 0
    raise ValueError(f"Invalid mode '{mode}'")


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

    if signal.shape[0] == 0:
        raise ValueError("signal cannot be empty")
    if kernel.shape[0] == 0:
        raise ValueError("kernel cannot be empty")

    signal, kernel, _signal_length, kernel_length = _order_longer_signal_first(signal, kernel)
    pad_left, pad_right = _convolution_1d_padding(mode, kernel_length)

    x = signal.reshape(1, 1, 1, -1)
    w = kernel.flip(0).reshape(1, 1, 1, -1)

    out = x.conv2d(w, padding=(pad_left, pad_right, 0, 0))
    return out.flatten()
