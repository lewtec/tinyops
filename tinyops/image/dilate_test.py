import cv2
import numpy as np
from tinygrad import Tensor, dtypes
from tinyops._core import assert_close
from tinyops.image.dilate import dilate

def test_dilate_grayscale():
    # Simple case: 5x5 image, 3x3 kernel
    img_np = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.uint8)

    kernel_np = np.ones((3, 3), dtype=np.uint8)

    # OpenCV result
    expected = cv2.dilate(img_np, kernel_np, iterations=1)

    # tinyops result
    img_tiny = Tensor(img_np)
    kernel_tiny = Tensor(kernel_np)
    result = dilate(img_tiny, kernel_tiny)

    assert_close(result, expected)

def test_dilate_random_grayscale():
    img_np = (np.random.rand(10, 10) * 255).astype(np.uint8)
    kernel_np = np.ones((3, 3), dtype=np.uint8)

    expected = cv2.dilate(img_np, kernel_np, iterations=1)

    img_tiny = Tensor(img_np)
    kernel_tiny = Tensor(kernel_np)
    result = dilate(img_tiny, kernel_tiny)

    assert_close(result, expected)

def test_dilate_color():
    # 3-channel image
    img_np = np.zeros((10, 10, 3), dtype=np.uint8)
    img_np[5, 5, :] = 255
    kernel_np = np.ones((3, 3), dtype=np.uint8)

    # Transpose for tinygrad's CHW format
    img_tiny_chw = Tensor(img_np.transpose(2, 0, 1))

    expected = cv2.dilate(img_np, kernel_np, iterations=1)

    kernel_tiny = Tensor(kernel_np)
    # The function expects CHW, so we pass it directly.
    result = dilate(img_tiny_chw, kernel_tiny)

    # Transpose result back to HWC for comparison
    assert_close(result.permute(1, 2, 0), expected)

def test_non_square_kernel():
    img_np = (np.random.rand(15, 15) * 255).astype(np.uint8)
    kernel_np = np.ones((5, 3), dtype=np.uint8)

    expected = cv2.dilate(img_np, kernel_np, iterations=1)

    img_tiny = Tensor(img_np)
    kernel_tiny = Tensor(kernel_np)
    result = dilate(img_tiny, kernel_tiny)

    assert_close(result, expected)

def test_complex_kernel():
    # Kernel with zeros
    img_np = np.zeros((10, 10), dtype=np.uint8)
    img_np[5, 5] = 100
    kernel_np = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    expected = cv2.dilate(img_np, kernel_np, iterations=1)

    img_tiny = Tensor(img_np)
    kernel_tiny = Tensor(kernel_np)
    result = dilate(img_tiny, kernel_tiny)

    assert_close(result, expected)
