## 2024-07-25 - Disabled Kernel Fusion Check (`assert_one_kernel`)
**Issue:** The `assert_one_kernel` decorator was present in the codebase but its logic was entirely commented out, making it misleading dead code.
**Root Cause:** The team has a directive to temporarily disable the single-kernel fusion check, but the implementation was left as commented-out code instead of being cleanly refactored.
**Solution:** I simplified the decorator into a clean, intentional no-op (pass-through) and removed the commented-out logic and redundant tests that verified its disabled state.
**Pattern:** The project has a known temporary architectural constraint: kernel fusion validation is disabled. Do not attempt to re-enable or test for it. When encountering related disabled code, the preferred solution is to refactor it into a clean no-op, not leave it as commented-out clutter.

## 2026-01-10 - Consolidate test utilities in _core module
**Issue:** Shared test utilities, specifically `assert_one_kernel`, were located in a `test_utils.py` file in the root `tinyops/` directory, while a `tinyops/_core/` directory existed for such shared code.
**Root Cause:** The project structure had evolved, but some older files were not moved to their more logical, centralized locations. This resulted in an inconsistent and slightly disorganized structure.
**Solution:** I moved `test_utils.py` and its corresponding test file, `test_utils_test.py`, into the `tinyops/_core/` directory. I then updated `tinyops/_core/__init__.py` to export the utility, ensuring no breaking changes to files that import it.
**Pattern:** All shared, internal utilities, whether for testing or runtime, should be consolidated within the `tinyops/_core/` module to maintain a clean and predictable project structure.
