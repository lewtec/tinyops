import cv2
import numpy as np
from tinygrad import dtypes, Tensor
from tinyops._core import assert_close
from tinyops.image.equalize_hist import equalize_hist

def test_equalize_hist_grayscale():
  # Create a random grayscale image
  gray_image = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)

  # Equalize the histogram using cv2
  equalized_image_cv2 = cv2.equalizeHist(gray_image)

  # Equalize the histogram using tinyops, ensuring the input is uint8
  gray_tensor = Tensor(gray_image, dtype=dtypes.uint8)
  equalized_image_tinyops = equalize_hist(gray_tensor)

  # Compare the results
  # Cast the output to float32 for comparison
  assert_close(equalized_image_tinyops.cast(dtypes.float32), equalized_image_cv2.astype(np.float32), atol=1, rtol=1e-5)
