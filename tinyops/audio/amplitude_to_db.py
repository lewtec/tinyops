from tinygrad import Tensor
import math

def amplitude_to_db(
    x: Tensor,
    stype: str = "power",
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> Tensor:
    """
    Turn a spectrogram from the amplitude/power scale to the decibel scale.

    Args:
        x: Input spectrogram.
        stype: Scale of the input spectrogram ("power" or "magnitude").
        ref: Reference value.
        amin: Minimum value to avoid log10(0).
        top_db: If not None, clamps the output to a minimum of max(output) - top_db.

    Returns:
        Tensor: Spectrogram in decibel scale.
    """
    if stype == "power":
        multiplier = 10.0
    elif stype == "magnitude":
        multiplier = 20.0
    else:
        raise ValueError("stype must be 'power' or 'magnitude'")

    ref_value = abs(ref)

    # Clamp the input to a minimum value
    x_clamped = x.maximum(amin)

    # Convert to decibels
    db_spec = multiplier * (x_clamped / ref_value).log() / math.log(10)

    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative")

        max_val = db_spec.max()
        db_spec = db_spec.maximum(max_val - top_db)

    return db_spec
