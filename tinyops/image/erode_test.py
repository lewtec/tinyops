import cv2
import numpy as np
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.erode import erode

def test_erode_grayscale():
    img_np = np.array([
        [255, 255, 255, 255, 255],
        [255, 100, 255, 255, 255],
        [255, 255, 255, 255, 255],
        [255, 255, 255, 100, 255],
        [255, 255, 255, 255, 255]
    ], dtype=np.uint8)

    kernel_np = np.ones((3, 3), dtype=np.uint8)

    expected = cv2.erode(img_np, kernel_np, iterations=1)

    img_tiny = Tensor(img_np)
    kernel_tiny = Tensor(kernel_np)
    result = erode(img_tiny, kernel_tiny)

    assert_close(result, expected)

def test_erode_random_grayscale():
    img_np = (np.random.rand(10, 10) * 255).astype(np.uint8)
    kernel_np = np.ones((3, 3), dtype=np.uint8)

    expected = cv2.erode(img_np, kernel_np, iterations=1)

    img_tiny = Tensor(img_np)
    kernel_tiny = Tensor(kernel_np)
    result = erode(img_tiny, kernel_tiny)

    assert_close(result, expected)

def test_erode_color():
    img_np = np.full((10, 10, 3), 255, dtype=np.uint8)
    img_np[5, 5, :] = 0
    kernel_np = np.ones((3, 3), dtype=np.uint8)

    img_tiny_chw = Tensor(img_np.transpose(2, 0, 1))

    expected = cv2.erode(img_np, kernel_np, iterations=1)

    kernel_tiny = Tensor(kernel_np)
    result = erode(img_tiny_chw, kernel_tiny)

    assert_close(result.permute(1, 2, 0), expected)

def test_non_square_kernel():
    img_np = (np.random.rand(15, 15) * 255).astype(np.uint8)
    kernel_np = np.ones((5, 3), dtype=np.uint8)

    expected = cv2.erode(img_np, kernel_np, iterations=1)

    img_tiny = Tensor(img_np)
    kernel_tiny = Tensor(kernel_np)
    result = erode(img_tiny, kernel_tiny)

    assert_close(result, expected)

def test_complex_kernel():
    img_np = np.full((10, 10), 255, dtype=np.uint8)
    img_np[5, 5] = 50
    kernel_np = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    expected = cv2.erode(img_np, kernel_np, iterations=1)

    img_tiny = Tensor(img_np)
    kernel_tiny = Tensor(kernel_np)
    result = erode(img_tiny, kernel_tiny)

    assert_close(result, expected)
