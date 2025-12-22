import cv2
import numpy as np
import pytest
from tinygrad import dtypes, Tensor
from tinyops._core import assert_close
from tinyops.image.normalize import normalize, NORM_MINMAX
from tinyops.test_utils import assert_one_kernel

def _get_input(input_dtype, output_dtype):
    if input_dtype == np.uint8:
        image = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
    else:
        image = (np.random.rand(100, 200) * 255).astype(np.float32)
    return Tensor(image, dtype=output_dtype).realize(), image

@pytest.mark.parametrize("input_dtype, output_dtype", [
    (np.float32, dtypes.float32),
    (np.uint8, dtypes.uint8)
])
@assert_one_kernel
def test_normalize_minmax(input_dtype, output_dtype):
    alpha, beta = 0, 255
    tensor, image = _get_input(input_dtype, output_dtype)

    # Normalize the image using cv2
    normalized_image_cv2 = cv2.normalize(image, None, alpha, beta, NORM_MINMAX, dtype=cv2.CV_32F if output_dtype == dtypes.float32 else cv2.CV_8U)

    # Normalize the image using tinyops
    normalized_image_tinyops = normalize(tensor, alpha=alpha, beta=beta, norm_type=NORM_MINMAX).realize()

    assert normalized_image_tinyops.dtype == output_dtype

    # Compare the results
    assert_close(normalized_image_tinyops.cast(dtypes.float32), normalized_image_cv2.astype(np.float32), atol=1, rtol=1e-5)
