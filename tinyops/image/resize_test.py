import cv2
import numpy as np
from tinygrad import dtypes, Tensor
from tinyops._core import assert_close
from tinyops.image.resize import resize

def test_resize_downscale():
    # Create a sample image (e.g., 10x10)
    img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    img_tensor = Tensor(img, dtype=dtypes.uint8)

    # Define output size
    out_size = (5, 5)

    # Run tinyops resize
    resized_tinyops = resize(img_tensor, out_size)

    # Run cv2 resize for comparison
    resized_cv2 = cv2.resize(img, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)

    # Compare results
    # Note: There might be slight differences due to floating point arithmetic,
    # so we use a tolerance. The tolerance value might need adjustment.
    assert_close(resized_tinyops, Tensor(resized_cv2), atol=1, rtol=1e-6)

def test_resize_upscale():
    # Create a sample image
    img = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)
    img_tensor = Tensor(img, dtype=dtypes.uint8)

    # Define output size
    out_size = (10, 10)

    # Run tinyops resize
    resized_tinyops = resize(img_tensor, out_size)

    # Run cv2 resize for comparison
    resized_cv2 = cv2.resize(img, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)

    # Compare results
    assert_close(resized_tinyops, Tensor(resized_cv2), atol=1, rtol=1e-6)

def test_resize_different_aspect_ratio():
    # Create a sample image
    img = np.random.randint(0, 256, (10, 5, 3), dtype=np.uint8)
    img_tensor = Tensor(img, dtype=dtypes.uint8)

    # Define output size
    out_size = (5, 10)

    # Run tinyops resize
    resized_tinyops = resize(img_tensor, out_size)

    # Run cv2 resize for comparison
    resized_cv2 = cv2.resize(img, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)

    # Compare results
    assert_close(resized_tinyops, Tensor(resized_cv2), atol=1, rtol=1e-6)
