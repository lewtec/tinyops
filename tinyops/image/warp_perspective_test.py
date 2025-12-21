import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops.image.warp_perspective import warp_perspective
from tinyops._core import assert_close

def test_warp_perspective_identity():
    img = np.random.rand(10, 20, 3).astype(np.float32)
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    dsize = (20, 10)

    result = warp_perspective(Tensor(img), Tensor(M), dsize)
    expected = cv2.warpPerspective(img, M, dsize)

    assert_close(result, expected)

def test_warp_perspective_simple():
    img = np.random.rand(10, 20, 1).astype(np.float32)
    src_pts = np.float32([[0, 0], [19, 0], [0, 9], [19, 9]])
    dst_pts = np.float32([[5, 5], [15, 5], [5, 15], [15, 15]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts).astype(np.float32)
    dsize = (25, 20)

    result = warp_perspective(Tensor(img), Tensor(M), dsize)
    expected = cv2.warpPerspective(img, M, dsize)

    assert_close(result, expected.reshape(20, 25, 1), atol=1e-2, rtol=1e-2)
