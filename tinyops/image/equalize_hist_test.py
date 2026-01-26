import cv2
import numpy as np
from tinygrad import Tensor, dtypes

from tinyops._core import assert_close, assert_one_kernel
from tinyops.image.equalize_hist import equalize_hist


def _get_input():
    gray_image = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
    return Tensor(gray_image, dtype=dtypes.uint8).realize(), gray_image


@assert_one_kernel
def test_equalize_hist_grayscale():
    gray_tensor, gray_image = _get_input()

    # Equalize the histogram using cv2
    equalized_image_cv2 = cv2.equalizeHist(gray_image)

    # Equalize the histogram using tinyops, ensuring the input is uint8
    equalized_image_tinyops = equalize_hist(gray_tensor).realize()

    # Compare the results
    # Cast the output to float32 for comparison
    assert_close(
        equalized_image_tinyops.cast(dtypes.float32), equalized_image_cv2.astype(np.float32), atol=1, rtol=1e-5
    )
