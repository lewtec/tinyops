from tinygrad import Tensor

def time_mask(spectrogram: Tensor, time_mask_param: int, mask_value: float = 0.0, iid_masks: bool = False) -> Tensor:
    """
    Apply masking to a spectrogram in the time domain.
    Reference: torchaudio.transforms.TimeMasking

    Args:
        spectrogram (Tensor): The input spectrogram of shape `(..., freq, time)`.
        time_mask_param (int): Maximum possible length of the mask.
        mask_value (float): The value to fill in the masked region.
        iid_masks (bool): Whether to apply different masks to each example/channel in the batch.

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

    if iid_masks and spectrogram.ndim > 2:
        rand_shape = spectrogram.shape[:-2] + (1, 1)
    else:
        rand_shape = (1,) * spectrogram.ndim

    t = (Tensor.rand(*rand_shape) * time_mask_param).floor()
    t0 = (Tensor.rand(*rand_shape) * (num_timesteps - t)).floor()

    time_indices = Tensor.arange(num_timesteps).reshape(1, 1, num_timesteps)

    mask = (time_indices >= t0) & (time_indices < t0 + t)

    return spectrogram.where(mask.logical_not(), mask_value)
