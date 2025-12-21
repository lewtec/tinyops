import pytest
import numpy as np
from PIL import Image
from tinygrad import Tensor, dtypes
from tinyops.io.encode_bmp import encode_bmp
from tinyops._core import assert_close

def test_encode_bmp(tmp_path):
    width, height = 15, 25
    # Create a dummy image with a gradient
    r = np.linspace(0, 255, width, dtype=np.uint8)
    g = np.linspace(255, 0, width, dtype=np.uint8)
    b = np.arange(width, dtype=np.uint8) * 10

    row = np.stack([r, g, b], axis=1)
    image_data = np.tile(row, (height, 1, 1))

    tensor = Tensor(image_data, dtype=dtypes.uint8)
    filepath = tmp_path / "test_output.bmp"

    encode_bmp(str(filepath), tensor)

    # Read back and verify
    with Image.open(filepath) as img:
        read_data = np.array(img)

    assert_close(Tensor(read_data, dtype=dtypes.uint8), tensor)
