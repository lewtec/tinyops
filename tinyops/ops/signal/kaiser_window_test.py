"""Pure-tinygrad tests for the Kaiser window (no reference libraries)."""

from tinyops.ops.signal.kaiser_window import _modified_bessel_i0, kaiser_window


def test_empty_and_single():
    assert kaiser_window(0, 5).shape == (0,)
    assert kaiser_window(-3, 5).shape == (0,)
    single = kaiser_window(1, 14).numpy()
    assert single.shape == (1,)
    assert abs(float(single[0]) - 1.0) < 1e-6


def test_beta_zero_is_ones():
    window = kaiser_window(9, 0.0).numpy()
    assert window.shape == (9,)
    for value in window:
        assert abs(float(value) - 1.0) < 1e-5


def test_odd_length_peak_and_symmetry():
    window = kaiser_window(11, 8.6).numpy()
    assert abs(float(window[5]) - 1.0) < 1e-6
    for left, right in zip(window, window[::-1]):
        assert abs(float(left) - float(right)) < 1e-6


def test_i0_at_zero_and_positive():
    assert abs(_modified_bessel_i0(0.0) - 1.0) < 1e-12
    # I0 is even and > 1 for |x| > 0
    assert _modified_bessel_i0(1.0) > 1.0
    assert abs(_modified_bessel_i0(1.0) - _modified_bessel_i0(-1.0)) < 1e-12
    # Domain split continuity around 8
    near = _modified_bessel_i0(7.9)
    far = _modified_bessel_i0(8.1)
    assert near > 1.0 and far > near
