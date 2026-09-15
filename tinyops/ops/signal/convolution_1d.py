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

    reversed_kernel = kernel.flip(0)
    padded_signal = signal.pad(((kernel_length - 1, kernel_length - 1),))

    full_length = signal_length + kernel_length - 1
    segments = [(reversed_kernel * padded_signal[i : i + kernel_length]).sum() for i in range(full_length)]
    full_result = Tensor.stack(segments)

    if mode == ConvolutionMode.FULL:
        return full_result
    elif mode == ConvolutionMode.SAME:
        start = (kernel_length - 1) // 2
        return full_result[start : start + signal_length]
    elif mode == ConvolutionMode.VALID:
        start = kernel_length - 1
        end = start + (signal_length - kernel_length + 1)
        return full_result[start:end]
    else:
        raise ValueError(f"Invalid mode '{mode}'")
