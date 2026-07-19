"""Pure-tinygrad tests for real DFT and its inverse (no reference libraries)."""

from tinygrad import Tensor

from tinyops.ops.signal.inverse_real_discrete_fourier_transform import (
    inverse_real_discrete_fourier_transform,
)
from tinyops.ops.signal.real_discrete_fourier_transform import real_discrete_fourier_transform


def test_empty_signal():
    spectrum = real_discrete_fourier_transform(Tensor.zeros(0))
    assert spectrum.shape == (0, 2)
    recovered = inverse_real_discrete_fourier_transform(spectrum)
    assert recovered.shape == (0,)


def test_single_sample():
    spectrum = real_discrete_fourier_transform(Tensor([3.5]))
    assert spectrum.shape == (1, 2)
    assert abs(float(spectrum[0, 0].numpy()) - 3.5) < 1e-5
    assert abs(float(spectrum[0, 1].numpy())) < 1e-5
    recovered = inverse_real_discrete_fourier_transform(spectrum, length=1)
    assert abs(float(recovered.numpy()[0]) - 3.5) < 1e-5


def test_roundtrip_small_even_and_odd():
    # Keep lengths small: each Cooley–Tukey size triggers clang kernel compile.
    for length in (2, 3, 4, 5):
        samples = Tensor([float(index + 1) for index in range(length)])
        spectrum = real_discrete_fourier_transform(samples)
        assert spectrum.shape == (length // 2 + 1, 2)
        recovered = inverse_real_discrete_fourier_transform(spectrum, length=length)
        recovered_values = recovered.numpy()
        original = samples.numpy()
        assert recovered.shape == (length,)
        for left, right in zip(recovered_values, original):
            assert abs(float(left) - float(right)) < 1e-3


def test_default_length_assumes_even():
    samples = Tensor([1.0, 2.0, 3.0, 4.0])
    spectrum = real_discrete_fourier_transform(samples)
    recovered = inverse_real_discrete_fourier_transform(spectrum)
    assert recovered.shape == (4,)
    for left, right in zip(recovered.numpy(), samples.numpy()):
        assert abs(float(left) - float(right)) < 1e-3


def test_rejects_non_1d_input():
    try:
        real_discrete_fourier_transform(Tensor([[1.0, 2.0], [3.0, 4.0]]))
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert "1-D" in str(error)
