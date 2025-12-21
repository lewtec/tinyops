import cv2
import numpy as np
from tinygrad import Tensor
from tinyops.image.warp_affine import warp_affine
from tinyops._core import assert_close

def test_warp_affine_identity():
    img = np.random.rand(10, 20, 3).astype(np.float32)
    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    dsize = (20, 10)

    result = warp_affine(Tensor(img), Tensor(M), dsize)
    expected = cv2.warpAffine(img, M, dsize)

    assert_close(result, expected)

def test_warp_affine_translation():
    img = np.random.rand(10, 20, 1).astype(np.float32)
    M = np.array([[1, 0, 5], [0, 1, 3]], dtype=np.float32)
    dsize = (25, 13)

    result = warp_affine(Tensor(img), Tensor(M), dsize)
    expected = cv2.warpAffine(img, M, dsize)

    assert_close(result, expected.reshape(13, 25, 1))

def test_warp_affine_rotation():
    img = np.random.rand(15, 15, 3).astype(np.float32)
    center = (7, 7)
    angle = 45
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    dsize = (15, 15)

    result = warp_affine(Tensor(img), Tensor(M), dsize)
    expected = cv2.warpAffine(img, M, dsize)

    assert_close(result, expected, atol=1e-1, rtol=1e-1)

def test_warp_affine_grayscale():
    img = np.random.rand(10, 20).astype(np.float32)
    M = np.array([[1, 0, 2], [0, 1, 1]], dtype=np.float32)
    dsize = (22, 11)

    result = warp_affine(Tensor(img), Tensor(M), dsize)
    expected = cv2.warpAffine(img, M, dsize)

    assert_close(result, expected)

def test_warp_affine_border_value():
    img = np.ones((5, 5), dtype=np.float32)
    M = np.array([[1, 0, 2], [0, 1, 2]], dtype=np.float32)
    dsize = (7, 7)
    border_value = 0.5

    result = warp_affine(Tensor(img), Tensor(M), dsize, border_value=border_value)
    expected = cv2.warpAffine(img, M, dsize, borderValue=border_value)

    assert_close(result, expected)
