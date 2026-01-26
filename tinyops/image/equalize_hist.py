from tinygrad import Tensor, dtypes

from tinyops.stats import bincount


def equalize_hist(src: Tensor) -> Tensor:
    """
    Equalizes the histogram of a grayscale image.
    The input is expected to be a 2D grayscale image of type uint8.
    """
    if src.dtype != dtypes.uint8:
        raise ValueError("Input image for equalize_hist must be of type uint8.")
    if len(src.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    # 1. Compute the histogram of the image.
    # The image is flattened to a 1D tensor to compute the histogram.
    # 256 bins are used for an 8-bit grayscale image.
    hist = bincount(src.flatten(), minlength=256)

    # 2. Compute the cumulative distribution function (CDF).
    cdf = hist.cumsum(axis=0)

    # 3. Normalize the CDF.
    # The formula is: h(v) = round(((cdf(v) - cdf_min) / (M*N - cdf_min)) * (L-1))
    # where M*N is the total number of pixels, L is the number of intensity levels (256),
    # and cdf_min is the minimum non-zero value of the CDF.

    # Find the minimum non-zero value in the CDF
    cdf_min = cdf.ne(0).where(cdf, float("inf")).min()

    num_pixels = src.shape[0] * src.shape[1]

    # Normalize the CDF to the range [0, 255]
    cdf_normalized = ((cdf - cdf_min) / (num_pixels - cdf_min)) * 255

    # Round to the nearest integer
    cdf_rounded = cdf_normalized.round()

    # 4. Map the original pixel values to the equalized values.
    # The equalized image is created by looking up the new value for each pixel
    # from the rounded, normalized CDF.
    equalized_image = cdf_rounded[src.flatten().cast(dtypes.int32)].reshape(src.shape)

    return equalized_image.cast(dtypes.uint8)
