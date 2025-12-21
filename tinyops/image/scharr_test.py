import cv2
import numpy as np
from tinygrad import Tensor, dtypes
from tinyops._core import assert_close
from tinyops.image.scharr import scharr

def test_scharr_dx():
    img = np.random.rand(10, 20).astype(np.float32) * 255
    img = img.astype(np.uint8)

    result_tinyops = scharr(Tensor(img, dtype=dtypes.uint8), dx=1, dy=0)
    result_opencv = cv2.Scharr(img, ddepth=cv2.CV_32F, dx=1, dy=0)

    assert_close(result_tinyops.cast(dtypes.float32), result_opencv, atol=1e-5, rtol=1e-5)

def test_scharr_dy():
    img = np.random.rand(10, 20).astype(np.float32) * 255
    img = img.astype(np.uint8)

    result_tinyops = scharr(Tensor(img, dtype=dtypes.uint8), dx=0, dy=1)
    result_opencv = cv2.Scharr(img, ddepth=cv2.CV_32F, dx=0, dy=1)

    assert_close(result_tinyops.cast(dtypes.float32), result_opencv, atol=1e-5, rtol=1e-5)

def test_scharr_color():
    img = np.random.rand(10, 20, 3).astype(np.float32) * 255
    img = img.astype(np.uint8)

    result_tinyops = scharr(Tensor(img, dtype=dtypes.uint8), dx=1, dy=0)

    result_opencv = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        result_opencv[:,:,i] = cv2.Scharr(img[:,:,i], ddepth=cv2.CV_32F, dx=1, dy=0)

    assert_close(result_tinyops.cast(dtypes.float32), result_opencv, atol=1e-5, rtol=1e-5)

def test_scharr_batch():
    img = np.random.rand(5, 10, 20, 3).astype(np.float32) * 255
    img = img.astype(np.uint8)

    result_tinyops = scharr(Tensor(img, dtype=dtypes.uint8), dx=0, dy=1)

    result_opencv = np.zeros_like(img, dtype=np.float32)
    for i in range(5):
      for j in range(3):
        result_opencv[i,:,:,j] = cv2.Scharr(img[i,:,:,j], ddepth=cv2.CV_32F, dx=0, dy=1)

    assert_close(result_tinyops.cast(dtypes.float32), result_opencv, atol=1e-5, rtol=1e-5)
