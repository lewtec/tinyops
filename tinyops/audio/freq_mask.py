from tinygrad import Tensor


def freq_mask(
    spectrogram: Tensor,
    freq_mask_param: int,
    mask_value: float = 0.0,
    iid_masks: bool = False,
    _rand_values: tuple = None,
    _freq_indices: Tensor = None,
) -> Tensor:
    """
    Apply masking to a spectrogram in the frequency domain.
    Reference: torchaudio.transforms.FrequencyMasking

    Args:
        spectrogram (Tensor): The input spectrogram of shape `(..., freq, time)`.
        freq_mask_param (int): Maximum possible length of the mask.
        mask_value (float): The value to fill in the masked region.
        iid_masks (bool): Whether to apply different masks to each example/channel in the batch.
        _rand_values (tuple): Internal parameter for testing. Tuple of (rand1, rand2) tensors.
        _freq_indices (Tensor): Internal parameter for testing. Pre-computed frequency indices tensor.

    Returns:
        Tensor: The masked spectrogram.
    """
    if spectrogram.ndim < 2:
        raise ValueError("Input spectrogram must be at least 2D.")
    if not 0 <= freq_mask_param:
        raise ValueError("freq_mask_param must be non-negative.")

    num_freqs = spectrogram.shape[-2]
    if freq_mask_param > num_freqs:
        raise ValueError(f"freq_mask_param ({freq_mask_param}) must be <= num_freqs ({num_freqs})")

    if freq_mask_param == 0:
        return spectrogram

    if iid_masks and spectrogram.ndim > 2:
        rand_shape = spectrogram.shape[:-2] + (1, 1)
    else:
        rand_shape = (1,) * spectrogram.ndim

    # Generate or use provided random values for mask length and starting position
    if _rand_values is not None:
        rand1, rand2 = _rand_values
    else:
        rand1 = Tensor.rand(*rand_shape)
        rand2 = Tensor.rand(*rand_shape)

    f_val = (rand1 * freq_mask_param).floor()
    f0_val = (rand2 * (num_freqs - f_val)).floor()

    # Create or use provided frequency index tensor
    if _freq_indices is None:
        freq_idx_shape = [1] * spectrogram.ndim
        freq_idx_shape[-2] = num_freqs
        freq_indices = Tensor.arange(num_freqs, dtype=spectrogram.dtype).reshape(freq_idx_shape)
    else:
        freq_indices = _freq_indices

    # Create mask: True where we want to mask (between f0 and f0+f)
    mask = (freq_indices >= f0_val) & (freq_indices < f0_val + f_val)

    # Apply mask
    return spectrogram.where(mask.logical_not(), mask_value)
