import cv2
import numpy as np
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.blur import blur

def test_blur():
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    ksize = (5, 5)

    # tinyops
    tiny_img = Tensor(img)
    tiny_blur = blur(tiny_img, ksize)

    # cv2
    cv_img = img.squeeze(0).transpose(1, 2, 0)
    cv_blur = cv2.blur(cv_img, ksize)
    cv_blur = cv_blur.transpose(2, 0, 1)

    assert_close(tiny_blur.squeeze(0), cv_blur)
