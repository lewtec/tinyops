from tinygrad import Tensor, dtypes
from typing import Optional, Tuple, Union, List
from .bincount import bincount

def hist2d(x: Tensor, y: Tensor, bins: Union[int, List[int]] = 10, range: Optional[List[List[float]]] = None, density: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    x, y = x.flatten(), y.flatten()

    if isinstance(bins, int):
        bins_x, bins_y = bins, bins
    else:
        bins_x, bins_y = bins[0], bins[1]

    if range is None:
        if x.numel() == 0:
            range_x, range_y = [0.0, 1.0], [0.0, 1.0]
        else:
            range_x = [x.min().item(), x.max().item()]
            range_y = [y.min().item(), y.max().item()]
    else:
        range_x, range_y = list(range[0]), list(range[1])

    if range_x[0] == range_x[1]:
        range_x[0] -= 0.5
        range_x[1] += 0.5
    if range_y[0] == range_y[1]:
        range_y[0] -= 0.5
        range_y[1] += 0.5

    edges_x = Tensor.linspace(range_x[0], range_x[1], bins_x + 1)
    edges_y = Tensor.linspace(range_y[0], range_y[1], bins_y + 1)

    if x.numel() == 0:
        return Tensor.zeros(bins_x, bins_y), edges_x, edges_y

    width_x = (range_x[1] - range_x[0]) / bins_x
    width_y = (range_y[1] - range_y[0]) / bins_y

    # Calculate indices
    idx_x = ((x - range_x[0]) / width_x).floor().cast(dtypes.int32)
    idx_y = ((y - range_y[0]) / width_y).floor().cast(dtypes.int32)

    # Handle the right edge case where index can be `bins`
    idx_x = (x == range_x[1]).where(bins_x - 1, idx_x)
    idx_y = (y == range_y[1]).where(bins_y - 1, idx_y)

    # Create a mask for values within the specified range.
    mask = (x >= range_x[0]) & (x <= range_x[1]) & (y >= range_y[0]) & (y <= range_y[1])

    # Flatten 2D indices to 1D
    flat_indices = (idx_x * bins_y + idx_y).cast(dtypes.int32)

    total_bins = bins_x * bins_y
    # Map outliers (values outside the range) to an outlier bin.
    final_indices = mask.where(flat_indices, total_bins)

    # Compute histogram using bincount
    hist_flat = bincount(final_indices, minlength=total_bins + 1)
    h = hist_flat[:total_bins].reshape(bins_x, bins_y)

    if density:
        db = width_x * width_y
        if h.sum().item() > 0:
          h = h / (h.sum() * db)

    return h, edges_x, edges_y
