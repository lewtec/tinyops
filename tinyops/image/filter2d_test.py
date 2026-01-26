import cv2
import numpy as np
import pytest
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.image.filter2d import filter2d

IMG_GRAYSCALE = np.array(
    [
        [10, 20, 30, 40, 50],
        [60, 70, 80, 90, 100],
        [110, 120, 130, 140, 150],
        [160, 170, 180, 190, 200],
        [210, 220, 230, 240, 250],
    ],
    dtype=np.uint8,
).astype(np.float32)

KERNEL_SYMMETRIC = (
    np.array(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ],
        dtype=np.float32,
    )
    / 16.0
)

KERNEL_NON_SYMMETRIC = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ],
    dtype=np.float32,
)

IMG_COLOR = np.stack(
    [
        np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8),
        np.array([[100, 110, 120], [130, 140, 150], [160, 170, 180]], dtype=np.uint8),
        np.array([[190, 200, 210], [220, 230, 240], [250, 10, 20]], dtype=np.uint8),
    ],
    axis=-1,
).astype(np.float32)

KERNEL_ONES = np.ones((3, 3), dtype=np.float32) / 9.0

TEST_PARAMS = [
    (IMG_GRAYSCALE, KERNEL_SYMMETRIC),
    (IMG_GRAYSCALE, KERNEL_NON_SYMMETRIC),
    (IMG_COLOR, KERNEL_ONES),
]


@pytest.mark.parametrize("img,kernel", TEST_PARAMS)
@assert_one_kernel
def test_filter2d(img, kernel):
    tensor_img = Tensor(img).realize()
    tensor_kernel = Tensor(kernel).realize()

    result = filter2d(tensor_img, tensor_kernel).realize()
    expected = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

    assert_close(result, expected)
