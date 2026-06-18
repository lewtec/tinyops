"""Tests for SciPy compatibility layer.

Compares tinyops.compat.scipy against actual scipy.
"""

import numpy as np
from scipy.spatial.distance import hamming as scipy_hamming
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.compat import scipy as tsp


class TestHammingDistance:
    def test_identical(self):
        a = Tensor([1, 0, 1, 0, 1])
        b = Tensor([1, 0, 1, 0, 1])
        result = tsp.spatial.distance.hamming(a, b)
        a_np = a.numpy()
        b_np = b.numpy()
        expected = scipy_hamming(a_np, b_np)
        assert_close(result, np.float32(expected), atol=1e-5)

    def test_completely_different(self):
        a = Tensor([1, 1, 1, 1])
        b = Tensor([0, 0, 0, 0])
        result = tsp.spatial.distance.hamming(a, b)
        assert_close(result, np.float32(1.0), atol=1e-5)

    def test_partial(self):
        a = np.array([1, 0, 0, 1, 1, 0], dtype=np.float32)
        b = np.array([0, 0, 1, 1, 0, 0], dtype=np.float32)
        result = tsp.spatial.distance.hamming(Tensor(a), Tensor(b))
        expected = scipy_hamming(a, b)
        assert_close(result, np.float32(expected), atol=1e-5)

    def test_single_element(self):
        a = Tensor([1.0])
        b = Tensor([0.0])
        result = tsp.spatial.distance.hamming(a, b)
        assert_close(result, np.float32(1.0), atol=1e-5)

    def test_float_values(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        result = tsp.spatial.distance.hamming(Tensor(a), Tensor(b))
        expected = scipy_hamming(a, b)
        assert_close(result, np.float32(expected), atol=1e-5)