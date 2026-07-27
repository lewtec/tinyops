"""Percentile estimation with NumPy-compatible interpolation methods."""

from enum import Enum
from math import isnan

from tinygrad import Tensor, dtypes

# Treat virtual-index fractional parts below this as exact integers.
# tinygrad float math can leave residual epsilons that pure NumPy does not.
_INTEGER_INDEX_TOLERANCE = 1e-5


class PercentileMethod(Enum):
    """Interpolation methods matching ``numpy.percentile`` / ``numpy.quantile``."""

    INVERTED_CDF = "inverted_cdf"
    AVERAGED_INVERTED_CDF = "averaged_inverted_cdf"
    CLOSEST_OBSERVATION = "closest_observation"
    INTERPOLATED_INVERTED_CDF = "interpolated_inverted_cdf"
    HAZEN = "hazen"
    WEIBULL = "weibull"
    LINEAR = "linear"
    MEDIAN_UNBIASED = "median_unbiased"
    NORMAL_UNBIASED = "normal_unbiased"
    LOWER = "lower"
    HIGHER = "higher"
    MIDPOINT = "midpoint"
    NEAREST = "nearest"


_CONTINUOUS_ALPHA_BETA: dict[PercentileMethod, tuple[float, float]] = {
    PercentileMethod.INTERPOLATED_INVERTED_CDF: (0.0, 1.0),
    PercentileMethod.HAZEN: (0.5, 0.5),
    PercentileMethod.WEIBULL: (0.0, 0.0),
    PercentileMethod.LINEAR: (1.0, 1.0),
    PercentileMethod.MEDIAN_UNBIASED: (1.0 / 3.0, 1.0 / 3.0),
    PercentileMethod.NORMAL_UNBIASED: (0.375, 0.375),
}

_DISCRETE_METHODS = frozenset(
    {
        PercentileMethod.INVERTED_CDF,
        PercentileMethod.CLOSEST_OBSERVATION,
        PercentileMethod.LOWER,
        PercentileMethod.HIGHER,
        PercentileMethod.NEAREST,
    }
)


def _normalize_axis(axis: int | tuple[int, ...] | None, number_of_dimensions: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(number_of_dimensions))
    if isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)
    normalized: list[int] = []
    for ax in axes:
        if ax < 0:
            ax += number_of_dimensions
        if ax < 0 or ax >= number_of_dimensions:
            raise ValueError(f"Axis {ax} out of bounds for array of dimension {number_of_dimensions}")
        normalized.append(ax)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"repeated axis in {axis}")
    return tuple(normalized)


def _as_probability_list(
    percentages: float | list[float] | Tensor,
) -> tuple[list[float], bool]:
    """Normalize percentage inputs to a list of probabilities in [0, 1]."""
    is_scalar_query = False
    if isinstance(percentages, (int, float)):
        values = [float(percentages) / 100.0]
        is_scalar_query = True
    elif isinstance(percentages, list):
        values = [float(value) / 100.0 for value in percentages]
    elif isinstance(percentages, Tensor):
        if len(percentages.shape) == 0:
            values = [float(percentages.tolist()) / 100.0]
            is_scalar_query = True
        elif len(percentages.shape) == 1:
            values = [float(value) / 100.0 for value in percentages.tolist()]
        else:
            raise ValueError("percentages must be 1D or scalar")
    else:
        raise TypeError(f"Unsupported type for percentages: {type(percentages)}")

    for value in values:
        if value < 0.0 or value > 1.0 or isnan(value):
            raise ValueError("Percentiles must be in the range [0, 100]")
    return values, is_scalar_query


def _virtual_indexes(
    sample_count: int,
    probabilities: Tensor,
    method: PercentileMethod,
) -> Tensor:
    n = float(sample_count)
    q = probabilities

    if method in _CONTINUOUS_ALPHA_BETA:
        alpha, beta = _CONTINUOUS_ALPHA_BETA[method]
        return n * q + (alpha + q * (1.0 - alpha - beta)) - 1.0

    if method == PercentileMethod.LOWER:
        return ((n - 1.0) * q).floor()
    if method == PercentileMethod.HIGHER:
        return ((n - 1.0) * q).ceil()
    if method == PercentileMethod.NEAREST:
        return ((n - 1.0) * q).round()
    if method == PercentileMethod.MIDPOINT:
        scaled = (n - 1.0) * q
        return 0.5 * (scaled.floor() + scaled.ceil())
    if method == PercentileMethod.AVERAGED_INVERTED_CDF:
        return n * q - 1.0
    if method == PercentileMethod.INVERTED_CDF:
        index = n * q - 1.0
        previous = index.floor()
        fraction = index - previous
        take_previous = fraction < _INTEGER_INDEX_TOLERANCE
        return (previous + (1.0 - take_previous.float())).clip(0, sample_count - 1)
    if method == PercentileMethod.CLOSEST_OBSERVATION:
        index = n * q - 1.0 - 0.5
        previous = index.floor()
        fraction = index - previous
        take_previous = (fraction < _INTEGER_INDEX_TOLERANCE) & ((previous % 2) == 1)
        return (previous + (1.0 - take_previous.float())).clip(0, sample_count - 1)

    raise ValueError(f"Unknown percentile method: {method}")


def _gather_along_last_axis(sorted_tensor: Tensor, indices: Tensor) -> Tensor:
    query_count = indices.shape[0]
    free_shape = list(sorted_tensor.shape[:-1])
    expanded_sorted = sorted_tensor.unsqueeze(0).expand([query_count] + list(sorted_tensor.shape))
    target_shape = [query_count] + free_shape + [1]
    index_view = indices.reshape((query_count,) + (1,) * len(free_shape) + (1,)).expand(target_shape)
    return expanded_sorted.gather(-1, index_view).squeeze(-1)


def _interpolate(
    sorted_tensor: Tensor,
    probabilities: Tensor,
    method: PercentileMethod,
) -> Tensor:
    sample_count = sorted_tensor.shape[-1]
    if sample_count == 0:
        raise ValueError("zero-size array to reduction operation `percentile` which has no identity")

    virtual = _virtual_indexes(sample_count, probabilities, method)

    if method in _DISCRETE_METHODS:
        integer_indices = virtual.cast(dtypes.int32).clip(0, sample_count - 1)
        return _gather_along_last_axis(sorted_tensor, integer_indices)

    previous_float = virtual.floor()
    next_float = previous_float + 1.0
    gamma = virtual - previous_float

    above = virtual >= (sample_count - 1)
    below = virtual < 0
    last_index = float(sample_count - 1)
    previous_float = above.where(last_index, previous_float)
    next_float = above.where(last_index, next_float)
    previous_float = below.where(0.0, previous_float)
    next_float = below.where(0.0, next_float)
    previous_float = previous_float.clip(0, sample_count - 1)
    next_float = next_float.clip(0, sample_count - 1)

    if method == PercentileMethod.AVERAGED_INVERTED_CDF:
        is_exact = gamma < _INTEGER_INDEX_TOLERANCE
        gamma = is_exact.where(0.5, 1.0)

    previous_values = _gather_along_last_axis(sorted_tensor, previous_float.cast(dtypes.int32))
    next_values = _gather_along_last_axis(sorted_tensor, next_float.cast(dtypes.int32))
    free_rank = len(sorted_tensor.shape) - 1
    gamma = gamma.reshape((probabilities.shape[0],) + (1,) * free_rank)
    return previous_values + (next_values - previous_values) * gamma




def _weighted_inverted_cdf_1d(values: list[float], weights: list[float], probabilities: list[float]) -> list[float]:
    if any(isnan(value) for value in values) or any(isnan(weight) for weight in weights):
        return [float("nan")] * len(probabilities)
    if any(weight < 0 for weight in weights):
        raise ValueError("Weights must be non-negative.")
    total = sum(weights)
    if total == 0 or any(weight == float("inf") for weight in weights):
        raise ValueError("Weights included NaN, inf or were all zero.")

    paired = sorted(zip(values, weights), key=lambda item: item[0])
    sorted_values = [item[0] for item in paired]
    sorted_weights = [item[1] for item in paired]

    cumulative: list[float] = []
    running = 0.0
    for weight in sorted_weights:
        running += weight
        cumulative.append(running / total)

    adjusted = [-1.0 if value == 0.0 else value for value in cumulative]
    last_index = len(sorted_values) - 1
    results: list[float] = []
    for probability in probabilities:
        index = 0
        while index < len(adjusted) and adjusted[index] < probability:
            index += 1
        if index > last_index:
            index = last_index
        results.append(sorted_values[index])
    return results


def _normalize_row(row) -> list[float]:
    if isinstance(row, list):
        return [float(value) for value in row]
    return [float(row)]


def _weighted_percentile(
    data_last_axis: Tensor,
    weights: Tensor,
    probability_values: list[float],
    free_shape: list[int],
    data_original_shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> Tensor:
    sample_count = data_last_axis.shape[-1]

    if weights.shape == data_original_shape:
        weight_last_axis = _align_weights_to_data(weights, data_original_shape, axes, free_shape, sample_count)
    elif len(weights.shape) == 1 and weights.shape[0] == sample_count:
        weight_last_axis = weights.reshape((1,) * len(free_shape) + (sample_count,)).expand(free_shape + [sample_count])
    else:
        raise ValueError(
            f"weights shape {weights.shape} is incompatible with data shape {data_original_shape} for reduction"
        )

    flat_count = 1
    for size in free_shape:
        flat_count *= size
    if flat_count == 0:
        raise ValueError("zero-size array to reduction operation `percentile` which has no identity")

    flat_values = data_last_axis.reshape(flat_count, sample_count).tolist()
    flat_weights = weight_last_axis.reshape(flat_count, sample_count).tolist()
    if flat_count == 1 and not isinstance(flat_values[0], list):
        flat_values = [_normalize_row(flat_values)]
        flat_weights = [_normalize_row(flat_weights)]

    query_count = len(probability_values)
    output_rows: list[list[float]] = []
    for row_values, row_weights in zip(flat_values, flat_weights):
        output_rows.append(
            _weighted_inverted_cdf_1d(_normalize_row(row_values), _normalize_row(row_weights), probability_values)
        )

    transposed = [[output_rows[row][query] for row in range(flat_count)] for query in range(query_count)]
    return Tensor(transposed, dtype=dtypes.float32).reshape([query_count] + free_shape)


def _align_weights_to_data(
    weights: Tensor,
    data_original_shape: tuple[int, ...],
    axes: tuple[int, ...],
    free_shape: list[int],
    sample_count: int,
) -> Tensor:
    number_of_dimensions = len(data_original_shape)
    remaining_axes = [index for index in range(number_of_dimensions) if index not in axes]
    permutation = remaining_axes + list(axes)
    aligned = weights.permute(permutation)
    return aligned.reshape(free_shape + [sample_count])


def _restore_keepdims_shape(
    free_shape: list[int],
    axes: tuple[int, ...],
    number_of_dimensions: int,
    query_count: int | None,
) -> list[int]:
    restored: list[int] = [] if query_count is None else [query_count]
    free_iter = iter(free_shape)
    for dim in range(number_of_dimensions):
        if dim in axes:
            restored.append(1)
        else:
            restored.append(next(free_iter))
    return restored


def _prepare_reduction(
    tensor: Tensor,
    axis: int | tuple[int, ...] | None,
) -> tuple[Tensor, list[int], tuple[int, ...], int, tuple[int, ...]]:
    original_shape = tuple(tensor.shape)
    number_of_dimensions = len(original_shape)
    axes = _normalize_axis(axis, number_of_dimensions)
    remaining_axes = [index for index in range(number_of_dimensions) if index not in axes]
    permutation = remaining_axes + list(axes)
    tensor = tensor.permute(permutation)
    free_shape = [original_shape[index] for index in remaining_axes]
    sample_count = 1
    for index in axes:
        sample_count *= original_shape[index]
    tensor = tensor.reshape(free_shape + [sample_count])
    return tensor, free_shape, axes, number_of_dimensions, original_shape


def percentile(
    tensor: Tensor,
    percentages: float | list[float] | Tensor,
    axis: int | tuple[int, ...] | None = None,
    keep_dimensions: bool = False,
    method: PercentileMethod = PercentileMethod.LINEAR,
    weights: Tensor | None = None,
) -> Tensor:
    """Compute the q-th percentile along an axis (NumPy-compatible).

    Supports all NumPy interpolation methods, multi-axis reduction, NaN
    propagation, and weighted ``inverted_cdf``.
    """
    if not isinstance(method, PercentileMethod):
        raise TypeError(f"method must be PercentileMethod, got {type(method)}")

    probability_values, is_scalar_query = _as_probability_list(percentages)
    probabilities = Tensor(probability_values, dtype=dtypes.float32)
    prepared, free_shape, axes, number_of_dimensions, original_shape = _prepare_reduction(tensor, axis)

    if weights is not None:
        if method != PercentileMethod.INVERTED_CDF:
            raise ValueError(f"Only method 'inverted_cdf' supports weights. Got: {method.value}.")
        result = _weighted_percentile(
            prepared,
            weights,
            probability_values,
            free_shape,
            original_shape,
            axes,
        )
    else:
        # Detect NaNs before sort: tinygrad's sort does not preserve NaN order.
        has_nan = prepared.isnan().any(axis=-1)
        sorted_tensor, _ = prepared.sort()
        result = _interpolate(sorted_tensor, probabilities, method)
        # result is always (query, *free) here
        has_nan = has_nan.reshape((1,) + tuple(has_nan.shape)).expand(result.shape)
        result = has_nan.where(float("nan"), result)

    if is_scalar_query:
        result = result.squeeze(0)
        if keep_dimensions:
            result = result.reshape(
                _restore_keepdims_shape(free_shape, axes, number_of_dimensions, query_count=None)
            )
    elif keep_dimensions:
        result = result.reshape(
            _restore_keepdims_shape(
                free_shape, axes, number_of_dimensions, query_count=len(probability_values)
            )
        )

    return result
