import numpy as np
from tinygrad import Tensor
from tinyops.ml.standard_scaler import standard_scaler
from tinyops._core import assert_close
from tinyops.test_utils import assert_one_kernel
from sklearn.preprocessing import StandardScaler
import pytest

@pytest.mark.parametrize("shape, constant_column", [
    ((100, 10), None),  # Standard case with random data
    ((50, 5), 2),       # Case with a constant column (zero variance)
])
def test_standard_scaler(shape, constant_column):
    data_np = np.random.randn(*shape).astype(np.float32)
    if constant_column is not None:
        data_np[:, constant_column] = 1.0  # Set a column to a constant value

    data = Tensor(data_np)
    data.realize()

    @assert_one_kernel
    def run_scaler():
        result = standard_scaler(data)
        result.realize()
        return result

    result = run_scaler()

    expected = StandardScaler().fit_transform(data_np)
    assert_close(result, expected)
