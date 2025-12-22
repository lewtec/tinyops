from tinygrad import Tensor

def convolve(a: Tensor, v: Tensor, mode: str = "full") -> Tensor:
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    Args:
        a: First one-dimensional input tensor.
        v: Second one-dimensional input tensor.
        mode: {'full', 'valid', 'same'}, optional.
              'full': Returns the convolution at each point of overlap, with an output shape of (N+M-1,).
              'valid': Returns output where the two tensors fully overlap.
              'same': Returns output of length max(M, N).

    Returns:
        The convolved tensor.
    """
    if a.ndim != 1 or v.ndim != 1:
        raise ValueError("Input tensors must be one-dimensional.")

    n, m = a.shape[0], v.shape[0]

    if n == 0:
        raise ValueError("a cannot be empty")
    if m == 0:
        raise ValueError("v cannot be empty")

    # Swap to make 'a' the longer tensor for consistency
    if m > n:
        a, v, n, m = v, a, m, n

    # --- Core logic: compute full convolution, then slice for other modes ---
    v_rev = v.flip(0)
    a_padded = a.pad(((m - 1, m - 1),))

    full_conv_len = n + m - 1
    result_tensors = [(v_rev * a_padded[i:i+m]).sum() for i in range(full_conv_len)]

    full_conv = Tensor.stack(result_tensors)

    if mode == 'full':
        return full_conv

    if mode == 'same':
        start = (m - 1) // 2
        return full_conv[start : start + n]

    if mode == 'valid':
        start = m - 1
        end = start + (n - m + 1)
        return full_conv[start : end]

    raise ValueError(f"Invalid mode '{mode}', must be 'full', 'valid', or 'same'.")
