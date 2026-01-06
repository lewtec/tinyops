# AGENTS.md

## Objective
tinyops is a library of operations implemented purely in tinygrad. The goal is to generate fused and optimized kernels that can be exported to other languages/runtimes.
The restriction of using only tinygrad for implementations (with external libs only for compliance tests) ensures that the entire computation graph passes through tinygrad's kernel fusion system.

**Functions are stateless.** There are no train loops, stateful classes, or fit/predict interfaces. The user sets up the loop and manages state. Functions receive data and return results.

## Setup
```bash
# install mise if not present
if ! command -v mise &> /dev/null; then
    curl https://mise.run | sh
fi

# mise installs uv, uv installs deps
mise install
uv sync
```

## Structure
```
src/tinyops/
├── _core/          # internal helpers (types, tolerances, validation)
├── linalg/         # linear algebra
├── stats/          # statistics, histograms
├── image/          # image transformations
├── audio/          # audio processing
├── signal/         # signal processing
├── io/             # file encoder/decoder, if implementable in tinygrad (wav, bmp)
└── ml/             # machine learning algorithms (sklearn-like)
```

Each function in its own file with collocated test:
```
module/
├── __init__.py
├── func.py
└── func_test.py
```

## Implementation
- Single runtime dependency: `tinygrad`
- Receives/returns `tinygrad.Tensor`
- Mandatory Type hints
- No `**kwargs`
- Paths use `pathlib`

```python
# src/tinyops/stats/hist.py
from tinygrad import Tensor

def hist(x: Tensor, bins: int) -> Tensor:
    """Calculates tensor histogram."""
    ...
```

## Tests
- `*_test.py` file next to implementation
- Compare output with original lib (numpy, cv2, torch*, etc)
- Use `_core.assert_close` helper for tolerance

Example test for `stats.median`:

```python
# src/tinyops/stats/median_test.py
import numpy as np
from tinygrad import Tensor
from tinyops.stats.median import median
from tinyops._core import assert_close
import pytest

@pytest.mark.parametrize("shape", [(100,)])
def test_median(shape):
    data_np = np.random.randn(*shape).astype(np.float32)
    data = Tensor(data_np)
    result = median(data)

    expected = np.median(data_np)
    assert_close(result, expected)
```

## Adding new function
1. Choose next unchecked function in `CHECKLIST.md`
2. Consult original lib documentation to understand expected behavior
3. Create file in `src/tinyops/{module}/{func}.py`
4. Implement function using only `tinygrad`
5. Create `src/tinyops/{module}/{func}_test.py` comparing with original lib
6. Import in module's `__init__.py`
7. Run `mise run test -- -k {func}` to validate
8. Mark as `[x]` in `CHECKLIST.md`

Example for `stats.median`:

```bash
# 1. implementation
touch src/tinyops/stats/median.py
```

```python
# src/tinyops/stats/median.py
from tinygrad import Tensor

def median(x: Tensor, axis: int | None = None) -> Tensor:
    """Returns the median along the axis."""
    ...
```

```python
# src/tinyops/stats/median_test.py
import numpy as np
from tinygrad import Tensor
from tinyops.stats.median import median
from tinyops._core import assert_close
import pytest

@pytest.mark.parametrize("shape", [(100,)])
def test_median(shape):
    data_np = np.random.randn(*shape).astype(np.float32)
    data = Tensor(data_np)

    result = median(data)

    expected = np.median(data_np)
    assert_close(result, expected)
```

```python
# src/tinyops/stats/__init__.py
from .median import median
```

## Commands
```bash
mise run test           # run all tests
mise run test -- -k hist  # test only hist
```
