## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-23 - DoS in Polynomial Features Expansion

**Vulnerability:** The `polynomial_features` function in `tinyops/ml` was vulnerable to Denial of Service (DoS) via memory exhaustion. It allowed generating an unbounded number of feature combinations ($O(N^d)$) without validation, potentially leading to application crashes.

**Learning:** Combinatorial algorithms must always validate input parameters against a safe threshold before execution. Relying on implicit system memory limits is unsafe. Pre-calculating the expected output size using closed-form math (like `math.comb`) is an efficient way to validate without allocation.

**Prevention:** Implement strict input validation for all functions that generate combinatorial or exponentially growing outputs. Use `math.comb` to predict output size and enforce hard limits (e.g., 1,000,000 features).
