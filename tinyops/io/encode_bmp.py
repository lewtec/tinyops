import struct
from tinygrad import Tensor, dtypes
import numpy as np

def encode_bmp(path: str, tensor: Tensor):
    """
    Encodes a tensor into a BMP file.
    NOTE: This function is not implemented in pure tinygrad, as tinygrad is not suited for file I/O and byte manipulation.
    This implementation only supports uncompressed 24-bit BMP files from a uint8 tensor.
    """
    if tensor.dtype != dtypes.uint8:
        raise ValueError("Tensor must be of dtype uint8.")

    height, width, channels = tensor.shape
    if channels != 3:
        raise ValueError("Tensor must have 3 channels (RGB).")

    # BMP row padding
    row_stride = (width * 3 + 3) & ~3
    padding = row_stride - (width * 3)

    # BMP File Header
    file_size = 54 + row_stride * height
    file_header = b'BM' + struct.pack('<IHHI', file_size, 0, 0, 54)

    # DIB Header (BITMAPINFOHEADER)
    dib_header = struct.pack('<IiiHHIIiiII', 40, width, height, 1, 24, 0, row_stride * height, 2835, 2835, 0, 0)

    with open(path, 'wb') as f:
        f.write(file_header)
        f.write(dib_header)

        image_np = tensor.numpy()

        # Convert RGB to BGR and write pixel data with padding
        for i in range(height):
            row = image_np[height - 1 - i, :, ::-1] # BGR and reverse row order
            f.write(row.tobytes())
            f.write(b'\x00' * padding)
