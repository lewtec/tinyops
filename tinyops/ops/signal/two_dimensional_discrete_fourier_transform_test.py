"""Pure-tinygrad tests for 2D DFT and its inverse (no reference libraries)."""

from tinygrad import Tensor

from tinyops.ops.signal.inverse_two_dimensional_discrete_fourier_transform import (
    inverse_two_dimensional_discrete_fourier_transform,
)
from tinyops.ops.signal.two_dimensional_discrete_fourier_transform import (
    two_dimensional_discrete_fourier_transform,
)


def test_empty_height():
    empty = Tensor.zeros(0, 3, 2)
    result = two_dimensional_discrete_fourier_transform(empty)
    assert result.shape == (0, 3, 2)


def test_empty_width():
    empty = Tensor.zeros(2, 0, 2)
    result = two_dimensional_discrete_fourier_transform(empty)
    assert result.shape == (2, 0, 2)


def test_single_pixel():
    pixel = Tensor([[[4.0, 0.0]]])
    spectrum = two_dimensional_discrete_fourier_transform(pixel)
    assert spectrum.shape == (1, 1, 2)
    assert abs(float(spectrum[0, 0, 0].numpy()) - 4.0) < 1e-5
    assert abs(float(spectrum[0, 0, 1].numpy())) < 1e-5
    recovered = inverse_two_dimensional_discrete_fourier_transform(spectrum)
    assert abs(float(recovered[0, 0, 0].numpy()) - 4.0) < 1e-5


def test_roundtrip_small_shapes():
    # Keep sizes tiny: each 1D Cooley–Tukey size triggers clang kernel compile.
    for height, width in ((2, 2), (2, 3), (3, 2), (4, 2)):
        values = [
            [float(row_index * width + column_index + 1) for column_index in range(width)]
            for row_index in range(height)
        ]
        packed = Tensor([[[value, 0.0] for value in row] for row in values])
        spectrum = two_dimensional_discrete_fourier_transform(packed)
        assert spectrum.shape == (height, width, 2)
        recovered = inverse_two_dimensional_discrete_fourier_transform(spectrum)
        assert recovered.shape == (height, width, 2)
        original = packed.numpy()
        recovered_values = recovered.numpy()
        for row_index in range(height):
            for column_index in range(width):
                assert abs(
                    float(recovered_values[row_index, column_index, 0])
                    - float(original[row_index, column_index, 0])
                ) < 1e-3
                assert abs(float(recovered_values[row_index, column_index, 1])) < 1e-3


def test_rejects_wrong_rank():
    try:
        two_dimensional_discrete_fourier_transform(Tensor([[1.0, 0.0], [2.0, 0.0]]))
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert "(H, W, 2)" in str(error)
