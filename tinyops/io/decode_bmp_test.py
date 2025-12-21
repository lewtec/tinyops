import pytest
import numpy as np
from PIL import Image
from tinyops.io.decode_bmp import decode_bmp
from tinyops._core import assert_close

@pytest.fixture
def bmp_file(tmp_path):
    width, height = 10, 20
    # Create a dummy image with a gradient
    r = np.linspace(0, 255, width, dtype=np.uint8)
    g = np.linspace(255, 0, width, dtype=np.uint8)
    b = np.zeros(width, dtype=np.uint8)

    # Create an image row and tile it vertically
    row = np.stack([r, g, b], axis=1)
    image_data = np.tile(row, (height, 1, 1))

    img = Image.fromarray(image_data.astype(np.uint8), 'RGB')
    filepath = tmp_path / "test.bmp"
    img.save(filepath)
    return filepath, image_data

def test_decode_bmp(bmp_file):
    filepath, expected_data = bmp_file
    tensor = decode_bmp(str(filepath))

    assert_close(tensor, expected_data)
