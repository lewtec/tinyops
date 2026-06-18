from enum import Enum

from tinygrad import Tensor


class CorrelationMode(Enum):
    """Defines how much of the cross-correlation output to return."""
    VALID = "valid"
    SAME = "same"
    FULL = "full"


def cross_correlation(
    signal: Tensor,
    kernel: Tensor,
    mode: CorrelationMode = CorrelationMode.VALID,
) -> Tensor:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        signal: First 1D input tensor.
        kernel: Second 1D input tensor.
        mode: Output size mode (VALID, SAME, or FULL).

    Returns:
        Cross-correlation result tensor.

    Raises:
        ValueError: If inputs are not 1D.
    """
    if signal.ndim != 1 or kernel.ndim != 1:
        raise ValueError("signal and kernel must be 1-dimensional")

    signal_length = signal.shape[0]
    kernel_length = kernel.shape[0]

    if mode == CorrelationMode.VALID:
        if signal_length < kernel_length:
            return Tensor([])
        output_length = signal_length - kernel_length + 1
        padded_signal = signal
    elif mode == CorrelationMode.SAME:
        output_length = signal_length
        pad_left = (kernel_length - 1) // 2
        pad_right = kernel_length - 1 - pad_left
        padded_signal = Tensor.cat(Tensor.zeros(pad_left), signal, Tensor.zeros(pad_right))
    elif mode == CorrelationMode.FULL:
        output_length = signal_length + kernel_length - 1
        padding = kernel_length - 1
        padded_signal = Tensor.cat(Tensor.zeros(padding), signal, Tensor.zeros(padding))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    segments = [(padded_signal[i : i + kernel_length] * kernel).sum() for i in range(output_length)]

    if not segments:
        return Tensor([])

    return Tensor.stack(segments)
