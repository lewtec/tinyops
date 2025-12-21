from tinygrad import Tensor, dtypes
import numpy as np
import cv2
import pytest

from tinyops.image.morphology import morphology, MorphTypes
from tinyops._core import assert_close

@pytest.mark.parametrize("op, cv2_op", [
    (MorphTypes.MORPH_OPEN, cv2.MORPH_OPEN),
    (MorphTypes.MORPH_CLOSE, cv2.MORPH_CLOSE),
    (MorphTypes.MORPH_GRADIENT, cv2.MORPH_GRADIENT),
    (MorphTypes.MORPH_TOPHAT, cv2.MORPH_TOPHAT),
    (MorphTypes.MORPH_BLACKHAT, cv2.MORPH_BLACKHAT),
])
def test_morphology_ops(op, cv2_op):
    img = np.zeros((10, 10), dtype=np.float32)
    img[4:7, 4:7] = 1.0
    kernel = np.ones((3, 3), dtype=np.uint8)

    img_uint8 = (img * 255).astype(np.uint8)

    expected = cv2.morphologyEx(img_uint8, cv2_op, kernel)
    expected = expected.astype(np.float32) / 255.0

    result = morphology(Tensor(img), Tensor(kernel, dtype=dtypes.uint8), op)

    assert_close(result, expected)

@pytest.mark.parametrize("op, cv2_op", [
    (MorphTypes.MORPH_OPEN, cv2.MORPH_OPEN),
    (MorphTypes.MORPH_CLOSE, cv2.MORPH_CLOSE),
    (MorphTypes.MORPH_GRADIENT, cv2.MORPH_GRADIENT),
    (MorphTypes.MORPH_TOPHAT, cv2.MORPH_TOPHAT),
    (MorphTypes.MORPH_BLACKHAT, cv2.MORPH_BLACKHAT),
])
def test_morphology_color(op, cv2_op):
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[4:7, 4:7, :] = 1.0
    kernel = np.ones((3, 3), dtype=np.uint8)

    img_uint8 = (img * 255).astype(np.uint8)

    expected = cv2.morphologyEx(img_uint8, cv2_op, kernel)
    expected = expected.astype(np.float32) / 255.0

    img_chw = Tensor(img).permute(2, 0, 1)

    result = morphology(img_chw, Tensor(kernel, dtype=dtypes.uint8), op)

    result_hwc = result.permute(1, 2, 0)

    assert_close(result_hwc, expected)
