from typing import Optional
from tinygrad import Tensor

def amplitude_to_db(x: Tensor, stype: str = 'power', top_db: Optional[float] = None) -> Tensor:
    x = x.abs()

    if stype == 'power':
        multiplier = 10.0
        amin = 1e-10
    elif stype == 'magnitude':
        multiplier = 20.0
        amin = 1e-5
    else:
        raise ValueError("stype must be one of 'power' or 'magnitude'")

    ref_value = x.max()
    db_multiplier = ref_value.maximum(amin).log10()

    x_db = multiplier * x.maximum(amin).log10() - (multiplier * db_multiplier)

    if top_db is not None:
        x_db = x_db.maximum(x_db.max() - top_db)

    return x_db
