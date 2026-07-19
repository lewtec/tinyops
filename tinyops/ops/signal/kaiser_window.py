"""Kaiser window via modified Bessel function of the first kind (order 0)."""

import math

from tinygrad import Tensor

# Clenshaw/Chebyshev coefficients for I0 on [0, 8] (Cephes / NumPy).
# See Clenshaw NPL Math Tables vol. 5; Abramowitz & Stegun §9.8.
_MODIFIED_BESSEL_I0_CHEBYSHEV_A: tuple[float, ...] = (
    -4.41534164647933937950e-18,
    3.33079451882223809783e-17,
    -2.43127984654795469359e-16,
    1.71539128555513303061e-15,
    -1.16853328779934516808e-14,
    7.67618549860493561688e-14,
    -4.85644678311192946090e-13,
    2.95505266312963983461e-12,
    -1.72682629144155570723e-11,
    9.67580903537323691224e-11,
    -5.18979560163526290666e-10,
    2.65982372468238665035e-9,
    -1.30002500998624804212e-8,
    6.04699502254191894932e-8,
    -2.67079385394061173391e-7,
    1.11738753912010371815e-6,
    -4.41673835845875056359e-6,
    1.64484480707288970893e-5,
    -5.75419501008210370398e-5,
    1.88502885095841655729e-4,
    -5.76375574538582365885e-4,
    1.63947561694133579842e-3,
    -4.32430999505057594430e-3,
    1.05464603945949983183e-2,
    -2.37374148058994688156e-2,
    4.93052842396707084878e-2,
    -9.49010970480476444210e-2,
    1.71620901522208775349e-1,
    -3.04682672343198398683e-1,
    6.76795274409476084995e-1,
)

# Clenshaw/Chebyshev coefficients for I0 on (8, inf).
_MODIFIED_BESSEL_I0_CHEBYSHEV_B: tuple[float, ...] = (
    -7.23318048787475395456e-18,
    -4.83050448594418207126e-18,
    4.46562142029675999901e-17,
    3.46122286769746109310e-17,
    -2.82762398051658348494e-16,
    -3.42548561967721913462e-16,
    1.77256013305652638360e-15,
    3.81168066935262242075e-15,
    -9.55484669882830764870e-15,
    -4.15056934728722208663e-14,
    1.54008621752140982691e-14,
    3.85277838274214270114e-13,
    7.18012445138366623367e-13,
    -1.79417853150680611778e-12,
    -1.32158118404477131188e-11,
    -3.14991652796324136454e-11,
    1.18891471078464383424e-11,
    4.94060238822496958910e-10,
    3.39623202570838634515e-9,
    2.26666899049817806459e-8,
    2.04891858946906374183e-7,
    2.89137052083475648297e-6,
    6.88975834691682398426e-5,
    3.36911647825569408990e-3,
    8.04490411014108831608e-1,
)

# Domain split for the two Chebyshev expansions (matches NumPy/Cephes).
_MODIFIED_BESSEL_I0_NEAR_DOMAIN_LIMIT = 8.0


def _chebyshev_evaluation(argument: float, coefficients: tuple[float, ...]) -> float:
    """Evaluate a Chebyshev series with Clenshaw recurrence."""
    first = coefficients[0]
    previous = 0.0
    current = first
    for coefficient in coefficients[1:]:
        older = previous
        previous = current
        current = argument * previous - older + coefficient
    return 0.5 * (current - older)


def _modified_bessel_i0(argument: float) -> float:
    """Modified Bessel function of the first kind, order 0 (I0).

    Uses the Cephes/Clenshaw expansions also used by NumPy: one polynomial on
    ``[0, 8]`` and another on ``(8, inf)``, scaled by ``exp(|x|)``.
    """
    absolute_argument = abs(float(argument))
    if absolute_argument <= _MODIFIED_BESSEL_I0_NEAR_DOMAIN_LIMIT:
        chebyshev_argument = absolute_argument / 2.0 - 2.0
        return math.exp(absolute_argument) * _chebyshev_evaluation(
            chebyshev_argument, _MODIFIED_BESSEL_I0_CHEBYSHEV_A
        )
    chebyshev_argument = 32.0 / absolute_argument - 2.0
    return (
        math.exp(absolute_argument)
        * _chebyshev_evaluation(chebyshev_argument, _MODIFIED_BESSEL_I0_CHEBYSHEV_B)
        / math.sqrt(absolute_argument)
    )


def kaiser_window(length: int, beta: float) -> Tensor:
    """Generate a Kaiser window.

    The Kaiser window is a tapered cosine window shaped by the modified
    Bessel function of the first kind:

    ``w[n] = I0(beta * sqrt(1 - ((n - alpha) / alpha)^2)) / I0(beta)``

    with ``alpha = (length - 1) / 2`` and ``n`` in ``0 .. length - 1``.

    Args:
        length: Number of points in the window. If less than 1, returns an
            empty tensor. Length 1 returns ``[1]``.
        beta: Shape parameter. ``0`` is rectangular; larger values narrow the
            main lobe (typical starting values are around 14).

    Returns:
        Tensor of window samples, peak-normalized to one when ``length`` is odd.
    """
    if length < 1:
        return Tensor([])
    if length == 1:
        return Tensor.ones(1)

    alpha = (length - 1) / 2.0
    normalization = _modified_bessel_i0(beta)
    samples: list[float] = []
    for index in range(length):
        relative = (index - alpha) / alpha
        # Clamp the radicand against tiny negative values from floating point.
        under_sqrt = max(0.0, 1.0 - relative * relative)
        samples.append(_modified_bessel_i0(beta * math.sqrt(under_sqrt)) / normalization)
    return Tensor(samples)
