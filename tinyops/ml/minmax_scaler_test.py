import numpy as np
from tinygrad import Tensor
from tinyops.ml.minmax_scaler import minmax_scaler
from tinyops._core import assert_close
from tinyops._core import assert_one_kernel
from sklearn.preprocessing import MinMaxScaler
import pytest

@pytest.mark.parametrize("shape, feature_range", [
    ((100, 10), (0, 1)),
    ((50, 5), (-1, 1)),
])
def test_minmax_scaler(shape, feature_range):
    data_np = np.random.randn(*shape).astype(np.float32)
    data = Tensor(data_np)
    data.realize()

    @assert_one_kernel
    def run_scaler():
        result = minmax_scaler(data, feature_range)
        result.realize()
        return result

    result = run_scaler()

    expected = MinMaxScaler(feature_range=feature_range).fit_transform(data_np)
    assert_close(result, expected, atol=1e-6, rtol=1e-6)
