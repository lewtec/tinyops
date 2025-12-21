import cv2
import numpy as np
import pytest
from tinygrad import dtypes, Tensor
from tinyops._core import assert_close
from tinyops.image.normalize import normalize, NORM_MINMAX

@pytest.mark.parametrize("input_dtype, output_dtype", [
    (np.float32, dtypes.float32),
    (np.uint8, dtypes.uint8)
])
def test_normalize_minmax(input_dtype, output_dtype):
    alpha, beta = 0, 255
    # Create a random image
    if input_dtype == np.uint8:
        image = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
    else:
        image = (np.random.rand(100, 200) * 255).astype(np.float32)

    # Normalize the image using cv2
    normalized_image_cv2 = cv2.normalize(image, None, alpha, beta, NORM_MINMAX, dtype=cv2.CV_32F if output_dtype == dtypes.float32 else cv2.CV_8U)

    # Normalize the image using tinyops
    tensor = Tensor(image, dtype=output_dtype)
    normalized_image_tinyops = normalize(tensor, alpha=alpha, beta=beta, norm_type=NORM_MINMAX)

    assert normalized_image_tinyops.dtype == output_dtype

    # Compare the results
    assert_close(normalized_image_tinyops.cast(dtypes.float32), normalized_image_cv2.astype(np.float32), atol=1, rtol=1e-5)
