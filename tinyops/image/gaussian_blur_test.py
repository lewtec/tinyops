import cv2
import numpy as np
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.gaussian_blur import gaussian_blur

def test_gaussian_blur():
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    ksize = (5, 5)
    sigmaX = 1.5

    # tinyops
    tiny_img = Tensor(img)
    tiny_gaussian_blur = gaussian_blur(tiny_img, ksize, sigmaX)

    # cv2
    cv_img = img.squeeze(0).transpose(1, 2, 0)
    cv_gaussian_blur = cv2.GaussianBlur(cv_img, ksize, sigmaX)
    cv_gaussian_blur = cv_gaussian_blur.transpose(2, 0, 1)

    assert_close(tiny_gaussian_blur.squeeze(0), cv_gaussian_blur)
