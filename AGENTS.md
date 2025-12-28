# AGENTS.md

## Objective
tinyops is a library of operations implemented purely in tinygrad. The goal is to generate fused and optimized kernels that can be exported to other languages/runtimes.
The restriction of using only tinygrad for implementations (with external libs only for compliance tests) ensures that the entire computation graph passes through tinygrad's kernel fusion system.

**Functions are stateless.** There are no train loops, stateful classes, or fit/predict interfaces. The user sets up the loop and manages state. Functions receive data and return results.

## Setup
```bash
# 🛡️ Sentinel: Install mise using a secure package manager.
# The `curl | sh` method is insecure and can lead to RCE.
#
# macOS:
# brew install mise
#
# Debian/Ubuntu:
# sudo apt install -y mise
#
# For other platforms, see https://mise.jdx.dev/installing-mise.html

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
- **Kernel Validation (CRITICAL):** All operator tests must validate kernel fusing continuity and efficiency. For this, use the `@assert_one_kernel` decorator (from `tinyops.test_utils`) in your tests. The test must be constructed so that the operation under test, along with the result realization, generates **exactly one kernel**.
    - **Do NOT use fixtures** for data preparation. Use `@pytest.mark.parametrize` even if there is only one test case.
    - Instantiate and realize (`.realize()`) input tensors *before* defining the decorated inner function, or ensure they are constants that do not generate extra kernels during function execution.
    - The goal is to ensure there are no breaks in the computation graph that prevent full operation fusing.

### Handling Kernel Fusing in Complex Tests

**IMPORTANT**: When the function being tested needs injected values (like mocked random values or auxiliary tensors), use `@pytest.mark.parametrize` BEFORE the `@assert_one_kernel` decorator. This allows tensors to be created OUTSIDE the measured block.

**Recommended Pattern:**
```python
@pytest.mark.parametrize("input_tensor,aux_values", [
    (
        Tensor(np.ones((10, 20), dtype=np.float32)).realize(),  # Created before test
        Tensor(np.array([0.5, 0.2], dtype=np.float32)),        # Auxiliary values
    )
])
@assert_one_kernel
def test_operation(input_tensor, aux_values):
    # Now just use parameters, without creating new tensors
    result = operation(input_tensor, _aux=aux_values)
    assert result.sum().item() > 0
```

**What NOT to do:**
- ❌ Create tensors inside the function decorated with `@assert_one_kernel`
- ❌ Use fixtures (resolved inside the measured block)
- ❌ Call `.realize()` inside the decorated function (counts as extra kernel)
- ❌ Use helper functions that create tensors inside the test

**Known Limitation**: For operations that use `Tensor.arange()` or `Tensor.ones()` internally, it may be necessary to add optional parameters (prefixed with `_`) to inject these tensors in tests:
```python
def operation(x: Tensor, param: int, _indices: Tensor = None) -> Tensor:
    """Operation that needs indices.

    Args:
        x: Input tensor
        param: Operation parameter
        _indices: (Internal) Pre-computed indices for testing
    """
    if _indices is None:
        _indices = Tensor.arange(x.shape[0])  # Generates kernel in production
    # Uses _indices in operations...
    return result
```

```python
# src/tinyops/stats/hist_test.py
import numpy as np
from tinygrad import Tensor
from tinyops.stats.hist import hist
from tinyops._core import assert_close
from tinyops.test_utils import assert_one_kernel
import pytest

# Use parametrize for inputs, avoiding fixtures
@pytest.mark.parametrize("size, bins", [(100, 256)])
def test_hist(size, bins):
    # Setup: Create and realize inputs outside monitored scope
    # Important: Realizing inputs here ensures Load/Creation kernels don't count
    data_np = np.random.randn(size).astype(np.float32)
    x = Tensor(data_np)
    x.realize()

    @assert_one_kernel
    def run_kernel():
        result = hist(x, bins=bins)
        result.realize() # Realize result. Counter must be exactly 1.
        return result

    # Execute decorated function
    result = run_kernel()

    # Value validation
    expected = np.histogram(data_np, bins=bins)[0]
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
from tinyops.test_utils import assert_one_kernel
import pytest

@pytest.mark.parametrize("shape", [(100,)])
def test_median(shape):
    # Setup
    data_np = np.random.randn(*shape).astype(np.float32)
    data = Tensor(data_np)
    data.realize()

    @assert_one_kernel
    def run_median():
        result = median(data)
        result.realize()
        return result

    result = run_median()

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
