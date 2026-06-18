# AGENTS.md

## Objective
tinyops is a library of operations implemented purely in tinygrad. The goal is to generate fused and optimized kernels that can be exported to other languages/runtimes.

The restriction of using only tinygrad for implementations (external libs are allowed only inside tests) ensures that the entire computation graph passes through tinygrad's kernel fusion system.

**Functions are stateless.** There are no train loops, stateful classes, or fit/predict interfaces. The user sets up the loop and manages state. Functions receive data and return results.

## Architecture
The library has two layers that must never be inverted:

- **tinyops.ops** — The final, preferred public API.
  - Poetic licence: we design for clarity, domain modeling, and fusion opportunities.
  - Subpackages represent domains (or cross-cutting concerns when an algorithm naturally belongs in multiple):
    - `tinyops.ops.image`
    - `tinyops.ops.audio`
    - `tinyops.ops.statistics`
    - `tinyops.ops.signal`
    - `tinyops.ops.linear_algebra`
    - `tinyops.ops.text`
    - `tinyops.ops.io`
    - `tinyops.ops.machine_learning`
  - Never depends on `compat`. May reference sibling modules inside `ops` for DRY.

- **tinyops.compat** — Thin 1:1 compatibility shims.
  - Subpackages are named after the exact libraries/versions being imitated:
    - `tinyops.compat.numpy1`
    - `tinyops.compat.numpy2`
    - `tinyops.compat.scipy`
    - `tinyops.compat.opencv4`
    - `tinyops.compat.torchaudio`
    - `tinyops.compat.torchvision`
    - `tinyops.compat.sklearn`
    - `tinyops.compat.filterpy`
    - and so on as needed.
  - Every public item required by the CHECKLIST must be present with matching signature, semantics, constants, and return shapes.
  - Implementation is *always* a thin wrapper/delegation to the corresponding `tinyops.ops` function or class. No logic lives here.
  - `compat` packages may reference other `compat` packages for shared adapter helpers, but never the reverse.
  - API compatibility takes precedence over kernel fusion in this layer.

**Direction is one way only:** `compat` → `ops`. No exceptions.

CHECKLIST.md is the scope manifest. Treat unchecked items as the remaining work list.

## Guidelines
These rules are non-negotiable unless a specific exception is recorded with strong justification.

### tinyops.compat rules
- Must provide 1:1 functionality and API surface with the reference implementation being mimicked.
- Tests must cover the *entire* surface including all documented edge cases, error conditions, dtypes, shapes, and parameter combinations that the reference supports.
- When reference data or fixtures are missing or insufficient for edge cases, generate the necessary inputs:
  - Use the reference library itself inside the test file to produce expected values.
  - For multimodal data (images, audio, video), synthesize what you need. As a multimodal model you may generate images, use `image_to_video`, or run `ffmpeg` commands via the terminal to create test signals, speech, noise, etc.
- No actual algorithmic implementation inside `compat/*`. Pure delegation.
- Exact signature and behavioral compatibility is mandatory, even when it prevents kernel fusion or feels un-idiomatic for tinygrad.
- Every checklist item under a given compat target must have passing tests that also stress edge cases.

### tinyops.ops rules
- This is our designed API. Use actual domain modeling.
- **PyTorch functional / albumentations style**: Functions or (preferred) classes take configuration arguments and return a callable (or are themselves callable) that accepts tensor(s):
  ```python
  # Preferred style
  class Resize:
      def __init__(self, target_size: tuple[int, int], method: InterpolationMethod = ...): ...
      def __call__(self, image: Tensor) -> Tensor: ...

  # Or factory returning the applicator
  def resize(target_size: tuple[int, int], method: InterpolationMethod = ...) -> Callable[[Tensor], Tensor]:
      ...
  ```
- Follow widespread conventions for value meanings:
  - Quantities treated as losses or error metrics are lower-is-better. If a reference uses higher-is-better, invert the signal in the modeling.
- Subpackages are primarily domain-oriented but flexible when an algorithm is useful across domains.
- **Domain modeling discipline**:
  - **Forbidden**: Hardcoded magic numbers or values sprinkled in logic. All constants must be named and justified (enums or module-level constants).
  - **Avoid**:
    - `**kwargs` used as a general escape hatch.
    - Useless comments that restate code.
    - Documentation that merely repeats the user's prompt or request.
    - TLAs or abbreviations of any size in public names (use `InterpolationMethod.BILINEAR`, not `IM.BIL` or `Interp.BILINEAR`).
  - **Encouraged**:
    - Enums for all discrete choices.
    - Frozen dataclasses or simple configuration classes where they improve clarity.
    - Exhaustive type hints (including `Callable[[Tensor], Tensor]`).
    - Docstrings that explain *why* and *what edge behavior* exists, not just what the function does.
    - DRY achieved by following the "rule of three": when a third similar piece appears, extract the common logic (often into private helpers inside the module or a sibling `_` module).
- Never alias or shorten names inside `tinyops.ops`. Full descriptive names (`arithmetic_mean`, `standard_deviation`, `morphological_erode`, `discrete_fourier_transform`, `apply_threshold`, ...).
- Production code (`ops/` + internal) may depend only on `tinygrad` and the Python standard library.

### Universal rules
- Single runtime dependency: `tinygrad`. All reference libraries (numpy, opencv, scipy, sklearn, torchaudio, torchvision, filterpy, etc.) are strictly **dev/test-only** dependencies.
- 100% test coverage is the baseline. Any uncovered code requires an *extremely well justified* reason recorded in the commit or a comment.
- In-package references inside `ops` (and inside `compat`) are encouraged to avoid duplication.
- All public symbols are re-exported through the appropriate `__init__.py`.

## Package Structure
```
tinyops/
├── __init__.py
├── _core/                 # shared test utilities, tolerances (never imported by production code paths)
├── compat/
│   ├── numpy2/
│   │   ├── __init__.py    # exact np.* surface
│   │   └── numpy2_test.py # exhaustive, including edges
│   ├── opencv4/
│   │   ├── __init__.py
│   │   └── opencv4_test.py
│   ├── scipy/, sklearn/, torchaudio/, torchvision/, filterpy/, ...
│   └── ...
└── ops/
    ├── __init__.py
    ├── image/
    │   ├── resize.py
    │   └── ...            # direct ops tests optional; covered via compat + internal
    ├── audio/, statistics/, signal/, linear_algebra/, machine_learning/, text/, io/
    └── ...
```

Each operation lives in its own file:
- `ops/<domain>/<descriptive_name>.py`
- Companion `*_test.py` is **mandatory for compat** layers. For core `ops` modules, add direct tests when complex internal logic benefits from isolation (these tests must not import reference libraries).

## Implementation (ops layer)
- Receives and returns `tinygrad.Tensor` (or collections thereof).
- Mandatory type hints.
- Use `pathlib` for any filesystem paths.
- Prefer small, well-typed private helpers over copy-pasted logic once the rule-of-three is triggered.

Example desired modeling (ops):

```python
# tinyops/ops/image/resize.py
from enum import Enum
from typing import Callable
from tinygrad import Tensor

class InterpolationMethod(Enum):
    NEAREST_NEIGHBOR = "nearest_neighbor"
    BILINEAR = "bilinear"

class Resize:
    """Resize transform (albumentations/PyTorch style)."""
    def __init__(
        self,
        target_size: tuple[int, int],
        method: InterpolationMethod = InterpolationMethod.BILINEAR,
    ):
        self.target_size = target_size
        self.method = method

    def __call__(self, image: Tensor) -> Tensor:
        ...
```

The corresponding compat layer then maps the reference's calling convention and constants onto the above:

```python
# tinyops/compat/opencv4/__init__.py
from tinyops.ops.image.resize import Resize, InterpolationMethod as _IM

INTER_LINEAR = 1
_INTER_MAP = {INTER_LINEAR: _IM.BILINEAR, ...}

def resize(src: Tensor, dsize: tuple[int, int], interpolation: int = INTER_LINEAR) -> Tensor:
    method = _INTER_MAP[interpolation]
    return Resize(target_size=(dsize[1], dsize[0]), method=method)(src)  # note height/width swap if needed
```

## Tests
- Only files under `*_test.py` (and `libs_test.py`) may import reference libraries.
- Every compat test file must drive the *full* API of its target, including error paths and unusual inputs.
- Use `tinyops._core.assert_close` (or equivalent internal helpers) for floating point comparisons.
- For ops that are not (yet) exposed via a compat layer, still aim for high-quality direct tests using only tinygrad-generated data.
- Run coverage and treat red coverage as a failure signal unless explicitly waived.

Example skeleton for a compat test (exhaustive style):

```python
import cv2
import numpy as np
from tinygrad import Tensor
from tinyops.compat import opencv4 as tcv
from tinyops._core import assert_close
import pytest

def test_resize_all_edges():
    for h, w in [(1,1), (8,8), (9,17), (64,32)]:
        for interp in (tcv.INTER_NEAREST, tcv.INTER_LINEAR):
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8).astype(np.float32)
            got = tcv.resize(Tensor(img), (w//2, h//2), interpolation=interp)
            expected = cv2.resize(img, (w//2, h//2), interpolation=interp)
            assert_close(got, expected, atol=2)
```

## Adding new functionality
1. Pick the next unchecked item from `CHECKLIST.md`.
2. Study the reference library's documentation and source behavior for the exact signature, constants, return types, and edge cases.
3. Implement the core logic inside the appropriate `tinyops/ops/<domain>/<name>.py` using the callable/transform style, enums, and strict domain modeling. No reference imports.
4. Add or extend tests:
   - For the `ops` implementation, add a direct `*_test.py` (pure tinygrad) when warranted for internal branches.
   - Create/extend the corresponding `tinyops/compat/<lib>/<lib>_test.py` with exhaustive coverage of the emulated API and all edge cases. Generate synthetic data (including audio/video via ffmpeg or model generation tools) when necessary.
5. Wire the symbol through the ops `__init__.py` (and the compat `__init__.py` with the exact reference name/signature).
6. `mise run test -- -k <relevant>` (and full coverage run) must pass.
7. Mark the item `[x]` in `CHECKLIST.md`.

The ops implementation is the source of truth for behavior. The compat layer only translates the surface.

## Commands
```bash
mise run test                 # run all tests
mise run test -- -k resize    # run tests whose name contains "resize"
mise run test -- --cov        # with coverage (enforce 100%)
```
