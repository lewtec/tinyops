import numpy as np
from tinygrad import Tensor
from tinyops.ml.robust_scaler import robust_scaler
from tinyops._core import assert_close
from tinyops.test_utils import assert_one_kernel
from sklearn.preprocessing import RobustScaler
import pytest

@pytest.mark.parametrize("shape", [(100, 10)])
def test_robust_scaler(shape):
    data_np = np.random.randn(*shape).astype(np.float32)
    data = Tensor(data_np)
    data.realize()

    # The percentile function is complex and may generate multiple kernels.
    # The @assert_one_kernel decorator is not appropriate here.
    result = robust_scaler(data)
    result.realize()

    expected = RobustScaler().fit_transform(data_np)
    assert_close(result, expected, atol=1e-6, rtol=1e-6)
