## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-25 - Prevent DoS in OneHotEncoder

**Vulnerability:** The `onehot_encoder` function lacked validation for the number of unique categories, which could lead to Memory Exhaustion (DoS) when processing high-cardinality features. This would generate a massive dense tensor (`N_samples * N_categories`).

**Learning:** Operations that expand data dimensions based on input content (like one-hot encoding or text vectorization) inherently carry DoS risks. Safe defaults (e.g., `max_categories=5000`) are essential for library robustness, even if they introduce potential breaking changes for edge cases.

**Prevention:** When implementing feature extraction or expansion logic, always include a `max_...` parameter (like `max_categories` or `max_features`) and enforce a reasonable default limit to fail fast rather than crash.
