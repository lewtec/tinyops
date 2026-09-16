from tinygrad import Tensor, dtypes

from tinyops.ops.statistics.bin_count import bin_count

INTENSITY_LEVELS = 256


def histogram_equalization(image: Tensor) -> Tensor:
    """Equalize the histogram of a grayscale image.

    Maps pixel intensities so that the output histogram is approximately
    uniform.

    Args:
        image: 2D grayscale image tensor of dtype uint8.

    Returns:
        Histogram-equalized image tensor (uint8).

    Raises:
        ValueError: If the image is not 2D uint8.
    """
    if image.dtype != dtypes.uint8:
        raise ValueError("Input image must be of type uint8.")
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    intensity_counts = bin_count(image.flatten(), minimum_output_length=INTENSITY_LEVELS)
    cumulative_distribution = intensity_counts.cumsum(axis=0)

    minimum_nonzero_cdf = cumulative_distribution.ne(0).where(cumulative_distribution, float("inf")).min()
    total_pixels = image.shape[0] * image.shape[1]

    normalized_cdf = ((cumulative_distribution - minimum_nonzero_cdf) / (total_pixels - minimum_nonzero_cdf)) * (
        INTENSITY_LEVELS - 1
    )
    rounded_cdf = normalized_cdf.round()

    equalized = rounded_cdf[image.flatten().cast(dtypes.int32)].reshape(image.shape)
    return equalized.cast(dtypes.uint8)
