"""Tests for OpenCV 4.x compatibility layer.

Compares tinyops.compat.opencv4 against actual cv2.
"""

import cv2
import numpy as np
import pytest
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.compat import opencv4 as tcv

# ============================================================================
# Helpers
# ============================================================================


def _make_grayscale(height: int = 64, width: int = 64) -> np.ndarray:
    return np.random.randint(0, 256, (height, width), dtype=np.uint8)


def _make_color(height: int = 64, width: int = 64) -> np.ndarray:
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


# ============================================================================
# Resize
# ============================================================================


class TestResize:
    def test_nearest(self):
        img = _make_grayscale(32, 32)
        dsize = (16, 16)
        result = tcv.resize(Tensor(img), dsize, interpolation=tcv.INTER_NEAREST)
        expected = cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)
        assert result.shape == expected.shape
        assert_close(result, expected, atol=1)

    def test_linear_color(self):
        img = _make_color(32, 32)
        dsize = (48, 48)
        result = tcv.resize(Tensor(img.astype(np.float32)), dsize, interpolation=tcv.INTER_LINEAR)
        expected = cv2.resize(img.astype(np.float32), dsize, interpolation=cv2.INTER_LINEAR)
        assert result.shape == expected.shape
        assert_close(result, expected, atol=2)

    def test_upscale(self):
        img = _make_grayscale(8, 8)
        dsize = (24, 24)
        result = tcv.resize(Tensor(img.astype(np.float32)), dsize, interpolation=tcv.INTER_LINEAR)
        assert result.shape == (24, 24)

    def test_rectangular(self):
        img = _make_grayscale(20, 30)
        dsize = (60, 40)  # width=60, height=40
        result = tcv.resize(Tensor(img.astype(np.float32)), dsize, interpolation=tcv.INTER_LINEAR)
        assert result.shape == (40, 60)


# ============================================================================
# Rotate
# ============================================================================


class TestRotate:
    def test_90_clockwise(self):
        img = _make_grayscale(20, 30)
        result = tcv.rotate(Tensor(img), tcv.ROTATE_90_CLOCKWISE)
        expected = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        assert_close(result, expected)

    def test_180(self):
        img = _make_color(20, 30)
        result = tcv.rotate(Tensor(img), tcv.ROTATE_180)
        expected = cv2.rotate(img, cv2.ROTATE_180)
        assert_close(result, expected)

    def test_90_counterclockwise(self):
        img = _make_grayscale(20, 30)
        result = tcv.rotate(Tensor(img), tcv.ROTATE_90_COUNTERCLOCKWISE)
        expected = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        assert_close(result, expected)


# ============================================================================
# Flip
# ============================================================================


class TestFlip:
    def test_horizontal(self):
        img = _make_grayscale()
        result = tcv.flip(Tensor(img), 1)
        expected = cv2.flip(img, 1)
        assert_close(result, expected)

    def test_vertical(self):
        img = _make_color()
        result = tcv.flip(Tensor(img), 0)
        expected = cv2.flip(img, 0)
        assert_close(result, expected)

    def test_both(self):
        img = _make_grayscale()
        result = tcv.flip(Tensor(img), -1)
        expected = cv2.flip(img, -1)
        assert_close(result, expected)


# ============================================================================
# Blur / Filtering
# ============================================================================


class TestBlur:
    def test_3x3(self):
        img = _make_grayscale().astype(np.float32)
        result = tcv.blur(Tensor(img), (3, 3))
        expected = cv2.blur(img, (3, 3))
        # Interior pixels should match; borders differ due to padding mode
        assert_close(result.numpy()[1:-1, 1:-1], expected[1:-1, 1:-1], atol=1)

    def test_5x5(self):
        img = _make_grayscale().astype(np.float32)
        result = tcv.blur(Tensor(img), (5, 5))
        expected = cv2.blur(img, (5, 5))
        assert_close(result.numpy()[2:-2, 2:-2], expected[2:-2, 2:-2], atol=1)


class TestGaussianBlur:
    def test_basic(self):
        img = _make_grayscale().astype(np.float32)
        result = tcv.GaussianBlur(Tensor(img), (5, 5), sigmaX=1.0)
        expected = cv2.GaussianBlur(img, (5, 5), 1.0)
        # Interior pixels should match; borders differ due to padding mode
        assert_close(result.numpy()[2:-2, 2:-2], expected[2:-2, 2:-2], atol=1)

    def test_color(self):
        img = _make_color().astype(np.float32)
        result = tcv.GaussianBlur(Tensor(img), (3, 3), sigmaX=0.5)
        expected = cv2.GaussianBlur(img, (3, 3), 0.5)
        assert_close(result.numpy()[1:-1, 1:-1], expected[1:-1, 1:-1], atol=2)


class TestFilter2D:
    def test_custom_kernel(self):
        img = _make_grayscale().astype(np.float32)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        result = tcv.filter2D(Tensor(img), -1, Tensor(kernel))
        expected = cv2.filter2D(img, -1, kernel)
        assert_close(result.numpy()[1:-1, 1:-1], expected[1:-1, 1:-1], atol=2)


# ============================================================================
# Edge detection
# ============================================================================


class TestSobel:
    def test_dx(self):
        img = _make_grayscale().astype(np.float32)
        result = tcv.Sobel(Tensor(img), -1, 1, 0, ksize=3)
        expected = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        assert_close(result.numpy()[1:-1, 1:-1], expected[1:-1, 1:-1], atol=1)

    def test_dy(self):
        img = _make_grayscale().astype(np.float32)
        result = tcv.Sobel(Tensor(img), -1, 0, 1, ksize=3)
        expected = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        assert_close(result.numpy()[1:-1, 1:-1], expected[1:-1, 1:-1], atol=1)


class TestScharr:
    def test_dx(self):
        img = _make_grayscale().astype(np.float32)
        result = tcv.Scharr(Tensor(img), -1, 1, 0)
        expected = cv2.Scharr(img, cv2.CV_32F, 1, 0)
        assert_close(result.numpy()[1:-1, 1:-1], expected[1:-1, 1:-1], atol=1)

    def test_dy(self):
        img = _make_grayscale().astype(np.float32)
        result = tcv.Scharr(Tensor(img), -1, 0, 1)
        expected = cv2.Scharr(img, cv2.CV_32F, 0, 1)
        assert_close(result.numpy()[1:-1, 1:-1], expected[1:-1, 1:-1], atol=1)


class TestLaplacian:
    def test_basic(self):
        img = _make_grayscale().astype(np.float32)
        result = tcv.Laplacian(Tensor(img), -1, ksize=1)
        expected = cv2.Laplacian(img, cv2.CV_32F, ksize=1)
        assert_close(result.numpy()[1:-1, 1:-1], expected[1:-1, 1:-1], atol=1)


# ============================================================================
# Morphology
# ============================================================================


class TestErode:
    def test_basic(self):
        img = _make_grayscale().astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)
        result = tcv.erode(Tensor(img), Tensor(kernel))
        expected = cv2.erode(img.astype(np.float32), kernel)
        # Interior should match; borders may differ due to padding
        assert_close(result.numpy()[1:-1, 1:-1], expected[1:-1, 1:-1], atol=1)


class TestDilate:
    def test_basic(self):
        img = _make_grayscale().astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)
        result = tcv.dilate(Tensor(img), Tensor(kernel))
        expected = cv2.dilate(img.astype(np.float32), kernel)
        assert_close(result.numpy()[1:-1, 1:-1], expected[1:-1, 1:-1], atol=1)


class TestMorphologyEx:
    def test_opening(self):
        img = _make_grayscale().astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)
        result = tcv.morphologyEx(Tensor(img), tcv.MORPH_OPEN, Tensor(kernel))
        expected = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # Compound morphology amplifies border differences
        assert_close(result.numpy()[2:-2, 2:-2], expected[2:-2, 2:-2], atol=1)

    def test_closing(self):
        img = _make_grayscale().astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)
        result = tcv.morphologyEx(Tensor(img), tcv.MORPH_CLOSE, Tensor(kernel))
        expected = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        assert_close(result.numpy()[2:-2, 2:-2], expected[2:-2, 2:-2], atol=1)


# ============================================================================
# Threshold
# ============================================================================


class TestThreshold:
    def test_binary(self):
        img = _make_grayscale().astype(np.float32)
        ret_ours, result = tcv.threshold(Tensor(img), 127, 255, tcv.THRESH_BINARY)
        ret_cv, expected = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        assert ret_ours == ret_cv
        assert_close(result, expected)

    def test_binary_inv(self):
        img = _make_grayscale().astype(np.float32)
        _, result = tcv.threshold(Tensor(img), 127, 255, tcv.THRESH_BINARY_INV)
        _, expected = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        assert_close(result, expected)

    def test_truncate(self):
        img = _make_grayscale().astype(np.float32)
        _, result = tcv.threshold(Tensor(img), 127, 255, tcv.THRESH_TRUNC)
        _, expected = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
        assert_close(result, expected)

    def test_tozero(self):
        img = _make_grayscale().astype(np.float32)
        _, result = tcv.threshold(Tensor(img), 127, 255, tcv.THRESH_TOZERO)
        _, expected = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
        assert_close(result, expected)

    def test_tozero_inv(self):
        img = _make_grayscale().astype(np.float32)
        _, result = tcv.threshold(Tensor(img), 127, 255, tcv.THRESH_TOZERO_INV)
        _, expected = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
        assert_close(result, expected)


# ============================================================================
# Color conversion
# ============================================================================


class TestCvtColor:
    def test_bgr2gray(self):
        img = _make_color().astype(np.float32)
        result = tcv.cvtColor(Tensor(img), tcv.COLOR_BGR2GRAY)
        expected = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert_close(result, expected, atol=1)

    def test_unsupported_code(self):
        img = _make_color().astype(np.float32)
        with pytest.raises(NotImplementedError):
            tcv.cvtColor(Tensor(img), 999)


# ============================================================================
# Histogram equalization
# ============================================================================


class TestEqualizeHist:
    def test_basic(self):
        img = _make_grayscale()
        result = tcv.equalizeHist(Tensor(img))
        expected = cv2.equalizeHist(img)
        assert_close(result, expected, atol=2)


# ============================================================================
# Normalize
# ============================================================================


class TestNormalize:
    def test_minmax(self):
        img = np.random.randn(10, 10).astype(np.float32)
        result = tcv.normalize(Tensor(img), None, alpha=0, beta=1, norm_type=tcv.NORM_MINMAX)
        expected = np.zeros_like(img)
        cv2.normalize(img, expected, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        assert_close(result, expected, atol=1e-4)