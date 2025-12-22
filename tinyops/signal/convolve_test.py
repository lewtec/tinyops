import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.signal.convolve import convolve

@pytest.mark.parametrize("mode", ["full", "valid", "same"])
@pytest.mark.parametrize("a_len, v_len", [(10, 3), (3, 10), (10, 10), (5, 1), (1, 5)])
def test_convolve_non_empty(mode, a_len, v_len):
    a_np = np.random.randn(a_len).astype(np.float32)
    v_np = np.random.randn(v_len).astype(np.float32)
    a_tg = Tensor(a_np)
    v_tg = Tensor(v_np)

    result_tg = convolve(a_tg, v_tg, mode=mode)
    result_np = np.convolve(a_np, v_np, mode=mode).astype(np.float32)
    assert_close(result_tg, result_np)

def test_convolve_empty_inputs():
    empty_tg = Tensor([])
    non_empty_tg = Tensor([1., 2., 3.])

    with pytest.raises(ValueError, match="a cannot be empty"):
        convolve(empty_tg, non_empty_tg)

    with pytest.raises(ValueError, match="v cannot be empty"):
        convolve(non_empty_tg, empty_tg)

    with pytest.raises(ValueError, match="a cannot be empty"):
        convolve(empty_tg, empty_tg)
