import cv2
import numpy as np
import pytest
from tinygrad import dtypes, Tensor
from tinyops._core import assert_close
from tinyops.image.cvt_color import cvt_color, COLOR_BGR2GRAY

@pytest.mark.parametrize("input_dtype, output_dtype", [
    (np.uint8, dtypes.uint8),
    (np.float32, dtypes.float32)
])
def test_bgr2gray(input_dtype, output_dtype):
    # Create a random BGR image with the specified data type
    if input_dtype == np.uint8:
        bgr_image = np.random.randint(0, 256, size=(100, 200, 3), dtype=np.uint8)
    else:
        bgr_image = np.random.rand(100, 200, 3).astype(np.float32)

    # Convert to grayscale using cv2
    gray_image_cv2 = cv2.cvtColor(bgr_image, COLOR_BGR2GRAY)

    # Convert to tinygrad Tensor
    bgr_tensor = Tensor(bgr_image.astype(np.float32))
    if output_dtype == dtypes.uint8:
        bgr_tensor = bgr_tensor.cast(dtypes.uint8)

    # Convert to grayscale using tinyops
    gray_image_tinyops = cvt_color(bgr_tensor, COLOR_BGR2GRAY)

    # Verify the output data type
    assert gray_image_tinyops.dtype == output_dtype, (
        f"Expected output dtype {output_dtype}, but got {gray_image_tinyops.dtype}"
    )

    # Compare the results
    assert_close(gray_image_tinyops.cast(dtypes.float32), gray_image_cv2.astype(np.float32), atol=1, rtol=1e-5)
