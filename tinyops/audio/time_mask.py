from tinygrad import Tensor

def time_mask(spectrogram: Tensor, time_mask_param: int, mask_value: float = 0.0, iid_masks: bool = False, _rand_values: tuple = None, _time_indices: Tensor = None) -> Tensor:
    """
    Apply masking to a spectrogram in the time domain.
    Reference: torchaudio.transforms.TimeMasking

    Args:
        spectrogram (Tensor): The input spectrogram of shape `(..., freq, time)`.
        time_mask_param (int): Maximum possible length of the mask.
        mask_value (float): The value to fill in the masked region.
        iid_masks (bool): Whether to apply different masks to each example/channel in the batch.
        _rand_values (tuple): Internal parameter for testing. Tuple of (rand1, rand2) tensors.
        _time_indices (Tensor): Internal parameter for testing. Pre-computed time indices tensor.

    Returns:
        Tensor: The masked spectrogram.
    """
    if spectrogram.ndim < 2:
        raise ValueError("Input spectrogram must be at least 2D.")
    if not 0 <= time_mask_param:
        raise ValueError("time_mask_param must be non-negative.")

    num_timesteps = spectrogram.shape[-1]
    if time_mask_param > num_timesteps:
        raise ValueError(f"time_mask_param ({time_mask_param}) must be <= num_timesteps ({num_timesteps})")

    if time_mask_param == 0:
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

    # Explicitly expand to input shape to force kernel fusion
    rand1 = rand1.expand(spectrogram.shape)
    rand2 = rand2.expand(spectrogram.shape)

    t_val = (rand1 * time_mask_param).floor()
    t0_val = (rand2 * (num_timesteps - t_val)).floor()

    # Create or use provided time index tensor
    if _time_indices is None:
        time_idx_shape = [1] * spectrogram.ndim
        time_idx_shape[-1] = num_timesteps
        time_indices = Tensor.arange(num_timesteps, dtype=spectrogram.dtype).reshape(time_idx_shape)
    else:
        time_indices = _time_indices

    # Explicitly expand time_indices to input shape to force kernel fusion
    time_indices = time_indices.expand(spectrogram.shape)

    # Create mask: True where we want to mask (between t0 and t0+t)
    mask = (time_indices >= t0_val) & (time_indices < t0_val + t_val)

    # Apply mask
    return spectrogram.where(mask.logical_not(), mask_value)
