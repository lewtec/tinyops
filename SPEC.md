# SPEC

Architecture contract and agent playbook for **tinyops**.

This document is the source of truth for product law and for how agents implement work. `AGENTS.md` only points here and lists commands. `CHECKLIST.md` is the compat scope backlog. `README.md` is a short human overview.

## 0. Meta

### Purpose

- **Architecture (A):** what tinyops is — layers, dependencies, surfaces, non-goals.
- **Playbook (C):** how an agent implements the next checklist item and proves it done.

### Requirement language

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are to be interpreted as described in RFC 2119.

- **MUST** / **MUST NOT** — mandatory. Non-compliance means the change is not done.
- **SHOULD** / **SHOULD NOT** — default expectation; deviate only with a clear, local reason.
- **MAY** — optional.

Playbook steps are **procedure**, not requirements. Requirements use the keywords above.

### Related documents

| Document | Role |
|----------|------|
| `SPEC.md` | Law (A) + agent playbook (C) |
| `AGENTS.md` | Pointer to this file + commands only |
| `CHECKLIST.md` | Compat scope: which reference symbols exist or remain |
| `README.md` | Human project overview |

### No escape

If a **MUST** cannot be met, the agent **MUST NOT** ship the change, **MUST NOT** mark the checklist item done, and **MUST** stop and ask the maintainer. Agents **MUST NOT** invent waivers, partial done-ness, or redefined drop-in criteria.

---

## 1. Product

### Goal

tinyops is a library of operations implemented purely in **tinygrad**, so the full computation graph stays inside tinygrad (and can be fused and exported to other runtimes).

### Surfaces

- **`tinyops.ops`** is the **product**: the designed, preferred public API. Domain modeling and fusion-friendly structure live here.
- **`tinyops.compat`** is for **testing** against reference libraries and, optionally, **migration** for users coming from those libraries. It is not the long-term destination API.

### Fusion

Kernel fusion is an **aspiration**, not a hard gate today. Designs **SHOULD** remain fusion-friendly. Production code **MUST** keep math on tinygrad (no escaping to NumPy/OpenCV/etc. in production paths) so fusion remains possible. There is **no** requirement that tests assert a specific kernel count.

### Statelessness (layered)

- **`ops` MUST be stateless** with respect to training frameworks: no fit/predict loops, no hidden learned state owned by the library. Functions (and config-then-call callables) receive data and return results. The caller owns the loop and any parameters they want to reuse.
- **`compat` MAY** expose class-shaped or fit/transform-shaped APIs when that matches the reference. Those are facades; they **MUST** delegate real work to `ops` (see §4).

---

## 2. Architecture

### Layers

Two layers; direction is one way only:

```text
compat  →  ops
```

- **`ops` MUST NOT** import or depend on `compat`.
- **`compat` MUST** depend on `ops` for all algorithmic behavior.
- **`compat` packages MAY** reference other `compat` packages for shared adapter helpers.
- **`ops` modules MAY** reference sibling `ops` modules for DRY.

### No orphan ops

There is **no** free-floating production op. Every production symbol under `ops` **MUST** be reachable from:

1. a `compat` facade, or  
2. another `ops` symbol that is eventually reachable from a `compat` facade under test.

Coverage of `compat` is therefore expected to exercise `ops` **by consequence**.

### Domains

`ops` subpackages are domain-oriented (and may host cross-cutting algorithms when that is the natural home), including at least:

- `tinyops.ops.image`
- `tinyops.ops.audio`
- `tinyops.ops.statistics`
- `tinyops.ops.signal`
- `tinyops.ops.linear_algebra`
- `tinyops.ops.text`
- `tinyops.ops.io`
- `tinyops.ops.machine_learning`

### Compat packages

`compat` subpackages are named after the library (and major line) being imitated, for example:

- `tinyops.compat.numpy2`
- `tinyops.compat.opencv4`
- `tinyops.compat.scipy`
- `tinyops.compat.sklearn`
- `tinyops.compat.torchaudio`
- `tinyops.compat.torchvision`
- `tinyops.compat.filterpy`

New compat targets **MUST** follow the same naming idea and appear on `CHECKLIST.md` before work is treated as in-scope.

### Package layout

```text
tinyops/
├── __init__.py
├── _core/                 # test utilities only (MUST NOT be used by production logic)
├── compat/
│   ├── <lib>/
│   │   ├── __init__.py    # facade surface
│   │   └── <lib>_test.py  # exhaustive tests vs reference
│   └── ...
└── ops/
    ├── __init__.py
    ├── <domain>/
    │   ├── <descriptive_name>.py
    │   └── ...
    └── ...
```

- Each operation **SHOULD** live in its own file: `ops/<domain>/<descriptive_name>.py`.
- Private helpers **MAY** live in `_*.py` modules within a domain.
- Public symbols **MUST** be re-exported through the appropriate `__init__.py` files.

### Runtime dependencies

- The only production runtime dependency **MUST** be **tinygrad** (plus the Python standard library).
- Reference libraries (NumPy, OpenCV, SciPy, scikit-learn, torchaudio, torchvision, filterpy, …) **MUST** be **dev/test-only**.
- `tinyops._core` is for tests (comparison helpers, etc.). It **MAY** import reference libraries. Production `ops` / `compat` code **MUST NOT** rely on `_core` for product logic.

---

## 3. `ops` law

### Role

`ops` is **our** API: clear domain modeling, full descriptive names, structured for the tinygrad graph.

### I/O

- Public `ops` entry points **MUST** accept and return `tinygrad.Tensor` (or collections thereof), not NumPy arrays as the primary contract.
- Type hints **SHOULD** be exhaustive.

### State

- `ops` **MUST** be stateless as defined in §1.
- Configuration in `__init__` of a callable transform **MAY** be stored if it is immutable configuration (sizes, enums, flags), not fitted model state hidden from the caller.
- When fit-vs-apply naturally splits, implementations **SHOULD** prefer explicit parameter tensors or pure functions the caller can hold, rather than hidden mutable instance fields in `ops`.

### Style and modeling

| Rule | Force |
|------|--------|
| Full descriptive public names (no TLA / abbreviation soup in public API) | **MUST** |
| No bare magic numbers; named constants or enums | **MUST** |
| Production imports: only `tinygrad` and the stdlib | **MUST** |
| Callable / transform style (class with `__call__`, or factory returning a callable) | **SHOULD** |
| Enums for discrete choices | **SHOULD** (and **MUST** when replacing magic numbers) |
| Exhaustive type hints | **SHOULD** |
| Docstrings that explain *why* and edge behavior | **SHOULD** |
| DRY via rule of three | **SHOULD** |
| Avoid `**kwargs` as a general escape hatch | **SHOULD** |
| Frozen dataclasses / small config types | **MAY** |

Metrics and losses in `ops` **SHOULD** follow lower-is-better conventions; if a reference is higher-is-better, invert in the `ops` model when that improves consistency.

### Example shape (illustrative)

```python
# tinyops/ops/image/resize.py
from enum import Enum
from tinygrad import Tensor

class InterpolationMethod(Enum):
    NEAREST_NEIGHBOR = "nearest_neighbor"
    BILINEAR = "bilinear"

class Resize:
    """Resize transform (albumentations / PyTorch functional style)."""

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

---

## 4. `compat` law

### Role

`compat` is **their** API: names, constants, calling conventions, and semantics of the reference library, proven against that library under CI.

### Facade only

- `compat` **MUST NOT** contain algorithmic implementations (no custom filters, solvers, or numeric methods beyond adaptation).
- `compat` **MUST** delegate math to `ops`.
- `compat` **MAY**:
  - map names, constants, and argument order;
  - perform structural adaptation (layout, dtype casts, packing/unpacking return values);
  - sequence multiple pure `ops` calls.
- If matching the reference requires different **math**, that math **MUST** live in `ops` (additional entry point or explicit mode), not in the facade.

### Drop-in

- For every symbol on `CHECKLIST.md` that is marked done, the corresponding `compat` surface **MUST** be **drop-in** relative to the reference for that symbol’s full documented fidelity: same meaningful results, shapes, and (where the reference guarantees them) dtypes, for the parameters and behaviors that symbol is specified to support.
- Drop-in is proven by tests that compare **tinyops.compat** to the **reference library** on the same inputs, using the versions resolved in the project’s **dev dependencies / lockfile**. **CI is the source of truth.**
- Non-floats **MUST** match exactly where the reference is exact (integers, booleans, discrete codes, etc.).
- Floats **MUST** match the reference within shared comparison helpers (e.g. `tinyops._core.assert_close`); tolerances **SHOULD** stay tight by default. Looser tolerances **MUST** be justified at the assert site.

### Errors

- Invalid use **MUST** fail somehow (no silent wrong success).
- Matching reference exception **classes** **MAY** be used to keep the API feel 1:1; full message parity is **not** required.

### Scope depth

- **Which** APIs: **checklist-literal** only. Unlisted symbols are out of scope until the maintainer adds them.
- **How deep** once listed: **full fidelity** for that symbol (documented parameters, overloads, dtypes, and edge cases needed for true drop-in). A happy-path-only facade is **not** done.

### Data plane (I/O)

- **Target:** `compat` **SHOULD** accept and return **reference-native** types (e.g. NumPy arrays for NumPy/OpenCV-style APIs) and convert at the facade boundary to/from `ops` Tensors.
- **Transitional:** existing Tensor-in/Tensor-out facades are allowed until touched.
- When implementing or substantially rewriting a checklist item, agents **SHOULD** prefer the reference-native boundary unless the maintainer says otherwise for that task.
- `ops` remains Tensor-only (§3).

### Illustrative facade (transitional Tensor style still valid until migrated)

```python
# tinyops/compat/opencv4/__init__.py
from tinyops.ops.image.resize import Resize, InterpolationMethod as _IM

INTER_LINEAR = 1
_INTER_MAP = {INTER_LINEAR: _IM.BILINEAR, ...}

def resize(src, dsize, interpolation=INTER_LINEAR):
    method = _INTER_MAP[interpolation]
    # map args, convert if needed, call ops, convert back if needed
    return Resize(target_size=(dsize[1], dsize[0]), method=method)(src)
```

---

## 5. Scope and done-ness

### CHECKLIST ownership

- The **maintainer** adds and removes rows on `CHECKLIST.md`.
- Agents **MUST NOT** expand compat scope by editing the checklist unless the maintainer explicitly asked for that.
- Agents implement **existing** unchecked items (or items the maintainer named in the task).

### Binary done-ness

- A checklist item is either **done** or **not**. There is no partial checklist state.
- An item **MUST** be marked `[x]` only when:
  1. the `ops` implementation exists and is wired,
  2. the `compat` facade exposes the symbol with full fidelity for that item,
  3. tests prove drop-in against the reference,
  4. relevant CI (including compat coverage) passes.
- Agents mark `[x]` when sending the PR that completes the item.

### Coverage

- **`compat` MUST** have **100%** line coverage.
- CI **MUST** fail if `tinyops.compat` coverage is under 100%.
- `ops` is **not** a separate coverage quota; it is covered by consequence of facades and their tests (and the no-orphan-ops rule).

---

## 6. Testing

### Who may import reference libraries

- **Only** `*_test.py` files and `libs_test.py` (and test-only helpers under `_core`) **MAY** import reference libraries.
- Production `ops` and `compat` code **MUST NOT** import them.

### Compat tests

- Each compat package **MUST** have tests that drive the full checklist surface for that package, including edges needed for drop-in.
- Expected values **MUST** come from the reference library (or fixtures derived from it) on the same inputs.
- When fixtures are missing, tests **SHOULD** synthesize inputs (including multimodal data via tools/ffmpeg when needed).
- Use `tinyops._core.assert_close` (or equivalent project helpers) for floating comparisons.

### Ops-only tests

- Direct pure-tinygrad tests under `ops` **MAY** exist for complex internal branches.
- They **MUST NOT** import reference libraries.
- They do not replace the requirement that production ops remain reachable from compat tests.

### Example skeleton

```python
import cv2
import numpy as np
from tinygrad import Tensor
from tinyops.compat import opencv4 as tcv
from tinyops._core import assert_close

def test_resize_edges():
    for h, w in [(1, 1), (8, 8), (9, 17)]:
        img = np.random.randint(0, 256, (h, w), dtype=np.uint8).astype(np.float32)
        got = tcv.resize(Tensor(img), (w // 2, h // 2), interpolation=tcv.INTER_LINEAR)
        expected = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        assert_close(got, expected, atol=2)
```

(As facades move to NumPy I/O, drop the manual `Tensor(...)` wrap at the call site.)

---

## 7. Playbook (agent procedure)

Implement **one** checklist item (or a small coherent set the maintainer named) as follows:

1. **Pick scope** — An unchecked row on `CHECKLIST.md` (or the item named in the task). Do not invent new checklist rows.
2. **Study the reference** — Signature, constants, dtypes, shapes, edges, and documented failure modes for **full fidelity**.
3. **Implement in `ops`** — `tinyops/ops/<domain>/<descriptive_name>.py` (or extend an existing op). Tensor I/O. No reference imports. Stateless. Domain modeling per §3.
4. **Facade in `compat`** — Thin mapping only (§4). Prefer reference-native I/O when adding or rewriting. No algorithms in the shim.
5. **Re-export** — Wire public symbols through the relevant `__init__.py` files.
6. **Test** — Extend `tinyops/compat/<lib>/<lib>_test.py` with reference comparisons covering full fidelity for that symbol. Generate synthetic data if needed.
7. **Run checks**
   - `mise run test -- -k <relevant>`
   - full test + compat coverage gate (see § Commands)
8. **Mark done** — Set the checklist item to `[x]` in the same change set / PR that completes it.
9. **If blocked by a MUST** — Stop, leave `[ ]`, ask the maintainer. No escape (§0).

`ops` is the source of truth for computation. `compat` only translates the surface.

---

## 8. Conventions (SHOULD / MAY)

These support clarity; they do not override §3–§5 MUSTs.

- Prefer small private helpers over copy-paste once the rule of three triggers (`_*.py` siblings are fine).
- Use `pathlib` for filesystem paths.
- Avoid comments that only restate the code.
- Avoid documentation that only restates the task prompt.
- In-package references inside `ops` (and inside `compat` for adapters only) are encouraged to avoid duplication.

### Preferred ops call style

```python
class Resize:
    def __init__(self, target_size: tuple[int, int], method: InterpolationMethod = ...): ...
    def __call__(self, image: Tensor) -> Tensor: ...

# or
def resize(target_size: tuple[int, int], method: InterpolationMethod = ...) -> Callable[[Tensor], Tensor]:
    ...
```

---

## Commands

```bash
mise run test                              # full test suite (CI)
mise run test -- -k resize                 # subset by test name
mise run test -- --cov=tinyops.compat --cov-fail-under=100   # compat coverage gate
mise run lint                              # ruff
mise run ci                                # lint + test
```

Agents **MUST** treat a failing compat coverage gate the same as a failing test: the change is not done.

---

## Summary of hard rules

1. `ops` is the product; `compat` is test + migration facade.  
2. Dependencies flow `compat` → `ops` only.  
3. Facades do not implement algorithms.  
4. No orphan ops.  
5. Production code: tinygrad + stdlib only.  
6. Checklist is maintainer-owned compat scope; full fidelity; binary done.  
7. Drop-in proven vs reference under CI/dev deps.  
8. 100% coverage on `compat`, enforced in CI.  
9. No escape: if a MUST cannot be met, stop and ask.
