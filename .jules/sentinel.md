## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-18 - Prevent DoS in PolynomialFeatures

**Vulnerability:** `polynomial_features` generates combinatorial features without validating the output size. Large inputs (e.g., degree 10, 50 features) cause massive memory allocation (75+ billion features), crashing the process (DoS).

**Learning:** Combinatorial algorithms can easily exceed memory limits. Input validation must account for the *output* complexity, not just the input size.

**Prevention:** Use closed-form formulas (like `math.comb`) to calculate the expected resource usage before starting the operation. Enforce hard limits on output size.
