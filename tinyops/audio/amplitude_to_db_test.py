import numpy as np
import pytest
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.audio.amplitude_to_db import amplitude_to_db


@pytest.mark.parametrize(
    "shape, stype, top_db",
    [
        ((10, 20), "power", 80.0),
        ((10, 20), "magnitude", 80.0),
        ((10, 20), "power", None),
        ((5, 15, 25), "magnitude", 60.0),
    ],
)
def test_amplitude_to_db(shape, stype, top_db):
    # a = np.random.rand(*shape).astype(np.float32)
    # create deterministic input
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.prod(shape)

    # tinygrad
    tinygrad_out = amplitude_to_db(Tensor(a), stype=stype, top_db=top_db)

    # reference implementation
    if stype == "power":
        multiplier = 10.0
    else:
        multiplier = 20.0

    ref_value = 1.0
    amin = 1e-10

    a_clamped = np.maximum(a, amin)
    expected_out = multiplier * np.log10(a_clamped / ref_value)

    if top_db is not None:
        expected_out = np.maximum(expected_out, expected_out.max() - top_db)

    assert_close(tinygrad_out, expected_out, atol=1e-5, rtol=1e-5)
