import cv2
import numpy as np
from tinygrad import dtypes, Tensor
from tinyops._core import assert_close
from tinyops.image.flip import flip

def test_flip_vertical():
    img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    img_tensor = Tensor(img, dtype=dtypes.uint8)

    flipped_tinyops = flip(img_tensor, 0)
    flipped_cv2 = cv2.flip(img, 0)

    assert_close(flipped_tinyops, Tensor(flipped_cv2))

def test_flip_horizontal():
    img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    img_tensor = Tensor(img, dtype=dtypes.uint8)

    flipped_tinyops = flip(img_tensor, 1)
    flipped_cv2 = cv2.flip(img, 1)

    assert_close(flipped_tinyops, Tensor(flipped_cv2))

def test_flip_both():
    img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    img_tensor = Tensor(img, dtype=dtypes.uint8)

    flipped_tinyops = flip(img_tensor, -1)
    flipped_cv2 = cv2.flip(img, -1)

    assert_close(flipped_tinyops, Tensor(flipped_cv2))
