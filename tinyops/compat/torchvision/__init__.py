"""torchvision compatibility layer.

Provides torchvision.transforms-compatible class signatures that delegate to tinyops.ops.
"""

from tinygrad import Tensor

from tinyops.ops.image.center_crop import center_crop as _center_crop
from tinyops.ops.image.pad import pad_image as _pad_image, PaddingMode


class _Transforms:
    """Namespace mimicking torchvision.transforms."""

    class CenterCrop:
        """Crop the image at the center to the given size."""

        def __init__(self, size: int | tuple[int, int]):
            self.size = size

        def __call__(self, image: Tensor) -> Tensor:
            return _center_crop(image, output_size=self.size)

    class Pad:
        """Pad the image on all sides with the given value."""

        _MODE_MAP = {
            "constant": PaddingMode.CONSTANT,
            "reflect": PaddingMode.REFLECT,
        }

        def __init__(self, padding: int | tuple[int, ...], fill: float = 0, padding_mode: str = "constant"):
            self.padding = padding
            self.fill = fill
            self.padding_mode = padding_mode

        def __call__(self, image: Tensor) -> Tensor:
            mode = self._MODE_MAP[self.padding_mode]
            return _pad_image(image, padding=self.padding, fill_value=self.fill, padding_mode=mode)


transforms = _Transforms()