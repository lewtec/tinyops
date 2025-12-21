import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.morphology import morphology, MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT

@pytest.fixture
def sample_image_and_kernel():
    img_np = (np.random.rand(20, 20) * 255).astype(np.uint8)
    kernel_np = np.ones((3, 3), dtype=np.uint8)
    img_tiny = Tensor(img_np)
    kernel_tiny = Tensor(kernel_np)
    return img_np, kernel_np, img_tiny, kernel_tiny

@pytest.mark.parametrize("op, cv2_op", [
    (MORPH_OPEN, cv2.MORPH_OPEN),
    (MORPH_CLOSE, cv2.MORPH_CLOSE),
    (MORPH_GRADIENT, cv2.MORPH_GRADIENT),
    (MORPH_TOPHAT, cv2.MORPH_TOPHAT),
    (MORPH_BLACKHAT, cv2.MORPH_BLACKHAT),
])
def test_morphology_ops(sample_image_and_kernel, op, cv2_op):
    img_np, kernel_np, img_tiny, kernel_tiny = sample_image_and_kernel

    expected = cv2.morphologyEx(img_np, cv2_op, kernel_np)
    result = morphology(img_tiny, op, kernel_tiny)

    assert_close(result, expected, atol=1e-5, rtol=1e-5) # Increased tolerance
