import cv2
import numpy as np
from tinygrad import Tensor, dtypes
from tinyops._core import assert_close
from tinyops.image.laplacian import laplacian

def test_laplacian_ksize_1():
    img = np.random.rand(10, 20).astype(np.float32) * 255
    img = img.astype(np.uint8)

    result_tinyops = laplacian(Tensor(img, dtype=dtypes.uint8), ksize=1)
    result_opencv = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=1)

    assert_close(result_tinyops, result_opencv, atol=1e-5, rtol=1e-5)

def test_laplacian_color():
    img = np.random.rand(10, 20, 3).astype(np.float32) * 255
    img = img.astype(np.uint8)

    result_tinyops = laplacian(Tensor(img, dtype=dtypes.uint8), ksize=1)

    result_opencv = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        result_opencv[:,:,i] = cv2.Laplacian(img[:,:,i], ddepth=cv2.CV_32F, ksize=1)

    assert_close(result_tinyops, result_opencv, atol=1e-5, rtol=1e-5)

def test_laplacian_batch():
    img = np.random.rand(5, 10, 20, 3).astype(np.float32) * 255
    img = img.astype(np.uint8)

    result_tinyops = laplacian(Tensor(img, dtype=dtypes.uint8), ksize=1)

    result_opencv = np.zeros_like(img, dtype=np.float32)
    for i in range(5):
      for j in range(3):
        result_opencv[i,:,:,j] = cv2.Laplacian(img[i,:,:,j], ddepth=cv2.CV_32F, ksize=1)

    assert_close(result_tinyops, result_opencv, atol=1e-5, rtol=1e-5)
