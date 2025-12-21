from tinygrad import dtypes, Tensor

# Color conversion codes from OpenCV
COLOR_BGR2GRAY = 6

def cvt_color(src: Tensor, code: int) -> Tensor:
  """
  Converts an image from one color space to another.
  This implementation aims to be compatible with OpenCV's cvtColor function.
  """
  if code == COLOR_BGR2GRAY:
    if src.shape[-1] != 3:
      raise ValueError("Input image must have 3 channels for BGR to Grayscale conversion.")

    # OpenCV uses the formula: Y = 0.299*R + 0.587*G + 0.114*B
    # For BGR images, the channels are in the order B, G, R.
    b, g, r = src[..., 0], src[..., 1], src[..., 2]
    grayscale = 0.114 * b + 0.587 * g + 0.299 * r

    # Preserve the data type of the input tensor
    if src.dtype == dtypes.uint8:
      return grayscale.cast(dtypes.uint8)
    return grayscale

  raise NotImplementedError(f"Color conversion code {code} not implemented yet.")
