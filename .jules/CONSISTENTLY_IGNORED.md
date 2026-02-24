# Consistently Ignored Changes

This file lists patterns of changes that have been consistently rejected by human reviewers. All agents MUST consult this file before proposing a new change. If a planned change matches any pattern described below, it MUST be abandoned.

---

## IGNORE: Do not add input validation to the WAV encoder

**- Pattern:** Do not add input validation logic to the `tinyops/io/encode_wav.py` file. While the WAV *decoder* has validation for security reasons, similar changes to the *encoder* are consistently rejected.
**- Justification:** Multiple pull requests attempting to add symmetric validation to the WAV encoder have been closed. This indicates that the change, while potentially well-intentioned, is considered unwanted noise or unnecessary for the project's goals.
**- Files Affected:** `tinyops/io/encode_wav.py`

---

## IGNORE: Do not attempt to fix DoS vulnerabilities in the WAV decoder

**- Pattern:** Do not add validation to the `tinyops/io/decode_wav.py` file to address Denial of Service (DoS) vulnerabilities.
**- Justification:** A pull request (e.g., #142) attempting to fix a DoS vulnerability by adding header validation to the WAV decoder was rejected. This indicates a consistent pattern of rejecting any changes, including security fixes, to the WAV I/O modules.
**- Files Affected:** `tinyops/io/decode_wav.py`

---

## IGNORE: Janitorial agents modifying CI/Tooling config

**- Pattern:** "Arrumador" or "Janitor" PRs that include changes to `.github/workflows/*.yml`, `mise.toml`, or `pyproject.toml` (infrastructure/tooling).
**- Justification:** Janitorial tasks must be strictly focused on code cleanup (e.g., lint fixes, unused imports). Bundling infrastructure changes or new tooling configuration (e.g., adding linters, centralized error reporting) consistently leads to rejection (e.g., PRs #211, #209). These changes belong in dedicated "Ops" PRs.
**- Files Affected:** `.github/workflows/*.yml`, `mise.toml`, `pyproject.toml`

---

## IGNORE: Wrapping Enum partials in tuples

**- Pattern:** Refactoring Enums to wrap `functools.partial` values in tuples (e.g., `MEMBER = (partial(func),)`).
**- Justification:** Multiple PRs (e.g., #204, #200, #194) attempting to wrap partials in tuples to support Python 3.14 have been rejected. This suggests the project prefers the existing cleaner syntax `MEMBER = partial(func)` even if it has forward compatibility issues, or the specific implementation was considered invasive.
**- Files Affected:** `tinyops/image/*.py`

---

## IGNORE: Non-existent GitHub Action versions

**- Pattern:** Updating `actions/checkout` to `v5` or referencing other non-existent action versions.
**- Justification:** Agents often hallucinate newer versions of GitHub Actions. Always verify the latest version tag exists before updating. `actions/checkout@v5` does not exist (v4 is current). Rejected in PR #189.
**- Files Affected:** `.github/workflows/*.yml`

---

## IGNORE: Arbitrary Resource Limits in Library Code

**- Pattern:** Adding hardcoded constants (e.g., `MAX_OUTPUT_FEATURES`) to limit input/output sizes for "DoS protection".
**- Justification:** Libraries should not enforce arbitrary limits on users. Resource management is the application's responsibility. Rejections (e.g., PR #201) confirm this is considered unwanted noise.
**- Files Affected:** `tinyops/ml/*.py`, `tinyops/image/*.py`

---

## IGNORE: Subjective "Educational" Disclaimers

**- Pattern:** Adding docstrings stating the implementation is "primarily for educational purposes", "extremely inefficient", or "do not use for large-scale".
**- Justification:** These disclaimers can be interpreted as disparaging the codebase or adding unnecessary noise. Documentation should focus on usage and objective complexity (e.g., "O(N!)") without subjective labeling. PR #197 was rejected for this pattern.
**- Files Affected:** `tinyops/linalg/*.py`, `tinyops/ml/*.py`

---

## IGNORE: Unnecessary Dictionary Lookups for Small Enums

**- Pattern:** Replacing simple `if/else` checks with dictionary lookups for small sets of Enums (e.g., color conversion codes).
**- Justification:** For small, static mappings, introducing a dictionary lookup can be considered over-engineering or premature optimization that adds complexity without significant benefit. PR #196 was rejected for this.
**- Files Affected:** `tinyops/image/*.py`

---

## IGNORE: Enforcing Local Numpy Imports

**- Pattern:** Refactoring module-level `numpy` imports to local imports inside functions (e.g., `def func(): import numpy as np`).
**- Justification:** Mass refactoring of existing top-level imports to local imports across modules (e.g., `tinyops/ml`) is considered unwanted noise/churn, as seen in rejected PRs #206 and #202. While reducing import time is good, mass changes like this are disruptive.
**- Files Affected:** `tinyops/ml/*.py`, `tinyops/io/*.py`

---

## IGNORE: Replacing Numpy Validation with Tensor Operations

**- Pattern:** Replacing simple `numpy`-based input validation (e.g., `np.unique`, `np.sum`) with complex `tinygrad` tensor operations, especially for eager validation before graph construction.
**- Justification:** The project explicitly allows numpy for validation "as it happens before the graph computation". Replacing clear, standard numpy logic with more obscure or complex tensor operations for validation purposes is rejected (PR #206).
**- Files Affected:** `tinyops/ml/*.py`
