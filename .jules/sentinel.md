## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-23 - DoS Prevention in Polynomial Features

**Vulnerability:** The `polynomial_features` function in `tinyops/ml` lacked input validation, allowing users to request an excessively large number of features (combinatorial explosion), leading to Denial of Service (DoS) via memory exhaustion.

**Learning:** Functions that generate combinatorial outputs must always validate the expected output size against a safe limit before allocation or computation. `math.comb` provides an efficient way to pre-calculate the size.

**Prevention:** Enforce strict output limits on all generative functions (e.g., `MAX_OUTPUT_FEATURES = 1_000_000`). Add regression tests that assert these limits are enforced by checking for `ValueError` on large inputs.
