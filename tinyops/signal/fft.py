from tinygrad import Tensor
import math

def _complex_mul(a: Tensor, b: Tensor) -> Tensor:
    """ Helper for complex multiplication (a+bi) * (c+di) = (ac-bd) + (ad+bc)i """
    # a, b are tensors of shape [..., N, 2]
    real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return Tensor.stack(real, imag, dim=-1)

def _fft_cooley_tukey(x: Tensor) -> Tensor:
    """ Recursive Cooley-Tukey FFT implementation """
    N = x.shape[-2]

    if N == 1:
        return x

    # Recursion
    even = _fft_cooley_tukey(x[..., ::2, :])
    odd = _fft_cooley_tukey(x[..., 1::2, :])

    # Twiddle factors
    k = Tensor.arange(N // 2, dtype=x.dtype)
    angle = -2 * math.pi * k / N

    # Using euler's formula e^(j*x) = cos(x) + j*sin(x)
    twiddle = Tensor.stack(angle.cos(), angle.sin(), dim=-1)

    # Unsqueeze for broadcasting if batch dimension exists
    if len(x.shape) > 2:
        twiddle = twiddle.unsqueeze(0)

    # Combine results
    term = _complex_mul(twiddle, odd)

    res_first_half = even + term
    res_second_half = even - term

    return res_first_half.cat(res_second_half, dim=-2)

def fft(x: Tensor) -> Tensor:
    """
    Computes the 1-dimensional discrete Fourier Transform.
    This implementation uses a recursive Cooley-Tukey algorithm.
    The input is expected to be a tensor representing complex numbers,
    where the last dimension has size 2 (real and imaginary parts).
    If the input is real, it will be converted to complex.
    The input length will be padded to the next power of two.
    """
    # If real, convert to complex
    if x.shape[-1] != 2:
        x = Tensor.stack(x, Tensor.zeros_like(x), dim=-1)

    # Cooley-Tukey requires power of 2 length
    N = x.shape[-2]
    if (N & (N - 1)) != 0 or N == 0:
        next_pow_2 = 1 << (N - 1).bit_length()
        pad_width = next_pow_2 - N
        # pad format is ((ax_0_before, ax_0_after), (ax_1_before, ax_1_after), ...)
        # we are padding the second to last dimension
        pad_spec = [(0,0)] * (len(x.shape) - 2) + [(0, pad_width), (0,0)]
        x = x.pad(tuple(pad_spec))

    return _fft_cooley_tukey(x)
