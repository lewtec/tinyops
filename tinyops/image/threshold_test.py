import cv2
import numpy as np
from tinygrad import Tensor
from tinyops.image.threshold import threshold, THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV
from tinyops._core import assert_close

def test_threshold():
    # Create a random grayscale image
    src = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    src_tensor = Tensor(src.astype(np.float32))
    thresh = 127.0
    maxval = 255.0

    # Test THRESH_BINARY
    _, expected = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)
    result = threshold(src_tensor, thresh, maxval, THRESH_BINARY)
    assert_close(result, Tensor(expected.astype(np.float32)))

    # Test THRESH_BINARY_INV
    _, expected = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY_INV)
    result = threshold(src_tensor, thresh, maxval, THRESH_BINARY_INV)
    assert_close(result, Tensor(expected.astype(np.float32)))

    # Test THRESH_TRUNC
    _, expected = cv2.threshold(src, thresh, maxval, cv2.THRESH_TRUNC)
    result = threshold(src_tensor, thresh, maxval, THRESH_TRUNC)
    assert_close(result, Tensor(expected.astype(np.float32)))

    # Test THRESH_TOZERO
    _, expected = cv2.threshold(src, thresh, maxval, cv2.THRESH_TOZERO)
    result = threshold(src_tensor, thresh, maxval, THRESH_TOZERO)
    assert_close(result, Tensor(expected.astype(np.float32)))

    # Test THRESH_TOZERO_INV
    _, expected = cv2.threshold(src, thresh, maxval, cv2.THRESH_TOZERO_INV)
    result = threshold(src_tensor, thresh, maxval, THRESH_TOZERO_INV)
    assert_close(result, Tensor(expected.astype(np.float32)))
