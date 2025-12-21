import struct
from tinygrad import Tensor, dtypes
import numpy as np

def decode_bmp(path: str) -> Tensor:
    """
    Decodes a BMP file into a tensor.
    NOTE: This function is not implemented in pure tinygrad, as tinygrad is not suited for file I/O and byte manipulation.
    This implementation only supports uncompressed 24-bit BMP files.
    """
    with open(path, 'rb') as f:
        # BMP File Header
        file_type = f.read(2)
        if file_type != b'BM':
            raise ValueError("Not a valid BMP file.")

        f.read(8) # Skip file size, reserved bytes
        pixel_data_offset = struct.unpack('<I', f.read(4))[0]

        # DIB Header (BITMAPINFOHEADER)
        header_size = struct.unpack('<I', f.read(4))[0]
        if header_size < 40:
            raise ValueError("Unsupported BMP header format.")

        width = struct.unpack('<i', f.read(4))[0]
        height = struct.unpack('<i', f.read(4))[0]

        f.read(2) # Skip color planes
        bits_per_pixel = struct.unpack('<H', f.read(2))[0]

        if bits_per_pixel != 24:
            raise ValueError("Only 24-bit BMP files are supported.")

        compression_method = struct.unpack('<I', f.read(4))[0]
        if compression_method != 0:
            raise ValueError("Compressed BMP files are not supported.")

        f.seek(pixel_data_offset)

        # Pixel Data
        row_stride = (width * 3 + 3) & ~3
        pixel_data = np.frombuffer(f.read(), dtype=np.uint8)

        # BMP stores rows in reverse order and with padding
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            row_start = i * row_stride
            row_end = row_start + width * 3
            image[height - 1 - i, :, :] = pixel_data[row_start:row_end].reshape(width, 3)

    # BGR to RGB
    image = image[:, :, ::-1]

    return Tensor(image.copy(), dtype=dtypes.uint8)
