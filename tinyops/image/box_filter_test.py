import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.box_filter import box_filter

@pytest.mark.parametrize("normalize", [True, False])
def test_box_filter(normalize):
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    ksize = (5, 5)

    # tinyops
    tiny_img = Tensor(img)
    tiny_box_filter = box_filter(tiny_img, ksize, normalize=normalize)

    # cv2
    cv_img = img.squeeze(0).transpose(1, 2, 0)
    cv_box_filter = cv2.boxFilter(cv_img, -1, ksize, normalize=normalize)
    cv_box_filter = cv_box_filter.transpose(2, 0, 1)

    assert_close(tiny_box_filter.squeeze(0), cv_box_filter)
