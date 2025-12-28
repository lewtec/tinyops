## 2024-07-25 - Disabled Kernel Fusion Check (`assert_one_kernel`)
**Issue:** The `assert_one_kernel` decorator was present in the codebase but its logic was entirely commented out, making it misleading dead code.
**Root Cause:** The team has a directive to temporarily disable the single-kernel fusion check, but the implementation was left as commented-out code instead of being cleanly refactored.
**Solution:** I simplified the decorator into a clean, intentional no-op (pass-through) and removed the commented-out logic and redundant tests that verified its disabled state.
**Pattern:** The project has a known temporary architectural constraint: kernel fusion validation is disabled. Do not attempt to re-enable or test for it. When encountering related disabled code, the preferred solution is to refactor it into a clean no-op, not leave it as commented-out clutter.
