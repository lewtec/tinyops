from tinygrad import Tensor

from tinyops.stats.cov import cov


def corrcoef(x: Tensor, y: Tensor | None = None, rowvar: bool = True) -> Tensor:
    """
    Return Pearson product-moment correlation coefficients.
    """
    c = cov(x, y, rowvar)

    # Extract the diagonal elements, which are the variances
    d = c.diagonal()

    # Compute standard deviations
    stddev = d.sqrt()

    # Create the outer product of stddev to form the normalization matrix
    norm_matrix = stddev.unsqueeze(1) @ stddev.unsqueeze(0)

    # Divide the covariance matrix by the normalization matrix
    # Add a small epsilon to avoid division by zero
    corr = c / (norm_matrix + 1e-10)

    return corr
