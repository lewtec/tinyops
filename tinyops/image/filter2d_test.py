import cv2
import numpy as np
from tinygrad import Tensor
from tinyops.image.filter2d import filter2d
from tinyops._core import assert_close

def test_filter2d_grayscale():
    # Use a fixed, deterministic grayscale image
    img_np = np.array([
        [10, 20, 30, 40, 50],
        [60, 70, 80, 90, 100],
        [110, 120, 130, 140, 150],
        [160, 170, 180, 190, 200],
        [210, 220, 230, 240, 250],
    ], dtype=np.uint8)
    kernel_np = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], dtype=np.float32) / 16.0

    # Convert to float32 for processing
    img_f32 = img_np.astype(np.float32)

    # Apply tinyops filter2d
    result = filter2d(Tensor(img_f32), Tensor(kernel_np))

    # Apply OpenCV filter2D
    expected = cv2.filter2D(img_f32, -1, kernel_np, borderType=cv2.BORDER_REFLECT_101)

    assert_close(result, expected)

def test_filter2d_non_symmetric_kernel():
    # Use a fixed, deterministic grayscale image
    img_np = np.array([
        [10, 20, 30, 40, 50],
        [60, 70, 80, 90, 100],
        [110, 120, 130, 140, 150],
        [160, 170, 180, 190, 200],
        [210, 220, 230, 240, 250],
    ], dtype=np.uint8)
    kernel_np = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=np.float32)

    # Convert to float32 for processing
    img_f32 = img_np.astype(np.float32)

    # Apply tinyops filter2d
    result = filter2d(Tensor(img_f32), Tensor(kernel_np))

    # Apply OpenCV filter2D
    expected = cv2.filter2D(img_f32, -1, kernel_np, borderType=cv2.BORDER_REFLECT_101)

    assert_close(result, expected)

def test_filter2d_color():
    # Use a fixed, deterministic color image
    img_np = np.stack([
        np.array([
            [10, 20, 30], [40, 50, 60], [70, 80, 90]
        ], dtype=np.uint8),
        np.array([
            [100, 110, 120], [130, 140, 150], [160, 170, 180]
        ], dtype=np.uint8),
        np.array([
            [190, 200, 210], [220, 230, 240], [250, 10, 20]
        ], dtype=np.uint8)
    ], axis=-1)
    kernel_np = np.ones((3, 3), dtype=np.float32) / 9.0

    # Convert to float32 for processing
    img_f32 = img_np.astype(np.float32)

    # Apply tinyops filter2d
    result = filter2d(Tensor(img_f32), Tensor(kernel_np))

    # Apply OpenCV filter2D
    expected = cv2.filter2D(img_f32, -1, kernel_np, borderType=cv2.BORDER_REFLECT_101)

    assert_close(result, expected)
