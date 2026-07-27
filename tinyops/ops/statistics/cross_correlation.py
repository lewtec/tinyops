from enum import Enum

from tinygrad import Tensor


class CorrelationMode(Enum):
    """Defines how much of the cross-correlation output to return."""

    VALID = "valid"
    SAME = "same"
    FULL = "full"


def _pad_signal_for_correlation(
    signal: Tensor,
    kernel_length: int,
    mode: CorrelationMode,
) -> tuple[Tensor, int]:
    """Return (padded_signal, output_length) for the given correlation mode."""
    signal_length = signal.shape[0]

    if mode == CorrelationMode.VALID:
        if signal_length < kernel_length:
            return signal, 0
        return signal, signal_length - kernel_length + 1

    if mode == CorrelationMode.SAME:
        pad_left = (kernel_length - 1) // 2
        pad_right = kernel_length - 1 - pad_left
        padded_signal = Tensor.cat(Tensor.zeros(pad_left), signal, Tensor.zeros(pad_right))
        return padded_signal, signal_length

    if mode == CorrelationMode.FULL:
        padding = kernel_length - 1
        padded_signal = Tensor.cat(Tensor.zeros(padding), signal, Tensor.zeros(padding))
        return padded_signal, signal_length + kernel_length - 1

    raise ValueError(f"Unknown mode: {mode}")


def _sliding_dot_products(padded_signal: Tensor, kernel: Tensor, output_length: int) -> Tensor:
    """Dot-product each contiguous window of *kernel* length with *kernel*."""
    if output_length <= 0:
        return Tensor([])

    kernel_length = kernel.shape[0]
    segments = [(padded_signal[i : i + kernel_length] * kernel).sum() for i in range(output_length)]
    if not segments:
        return Tensor([])
    return Tensor.stack(segments)


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

    padded_signal, output_length = _pad_signal_for_correlation(signal, kernel.shape[0], mode)
    return _sliding_dot_products(padded_signal, kernel, output_length)
