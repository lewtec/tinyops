from tinygrad import Tensor

def freq_mask(spectrogram: Tensor, freq_mask_param: int, mask_value: float = 0.0, iid_masks: bool = False) -> Tensor:
    """
    Apply masking to a spectrogram in the frequency domain.
    Reference: torchaudio.transforms.FrequencyMasking

    Args:
        spectrogram (Tensor): The input spectrogram of shape `(..., freq, time)`.
        freq_mask_param (int): Maximum possible length of the mask.
        mask_value (float): The value to fill in the masked region.
        iid_masks (bool): Whether to apply different masks to each example/channel in the batch.

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

    if iid_masks and spectrogram.ndim > 2:
        rand_shape = spectrogram.shape[:-2] + (1, 1)
    else:
        rand_shape = (1,) * spectrogram.ndim

    f = (Tensor.rand(*rand_shape) * freq_mask_param).floor()
    f0 = (Tensor.rand(*rand_shape) * (num_freqs - f)).floor()

    freq_indices = Tensor.arange(num_freqs).reshape(1, num_freqs, 1)

    mask = (freq_indices >= f0) & (freq_indices < f0 + f)

    return spectrogram.where(mask.logical_not(), mask_value)
