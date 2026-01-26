import numpy as np
import pytest
from sklearn.preprocessing import MaxAbsScaler
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.ml.maxabs_scaler import maxabs_scaler


@pytest.mark.parametrize("shape", [(100, 10)])
def test_maxabs_scaler(shape):
    data_np = np.random.randn(*shape).astype(np.float32)
    data = Tensor(data_np)
    data.realize()

    @assert_one_kernel
    def run_scaler():
        result = maxabs_scaler(data)
        result.realize()
        return result

    result = run_scaler()

    expected = MaxAbsScaler().fit_transform(data_np)
    assert_close(result, expected, atol=1e-6, rtol=1e-6)
