## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-19 - DoS in Polynomial Features

**Vulnerability:** The `polynomial_features` function lacked input validation for the expected number of output features. A malicious input with high `degree` or `n_features` could cause a Denial of Service (DoS) via memory exhaustion by generating an astronomical number of combinations.

**Learning:** Combinatorial functions must always pre-calculate the expected output size using cheap arithmetic operations (like `math.comb`) before attempting to allocate memory or generate the data. This "look-before-you-leap" pattern is critical for preventing resource exhaustion attacks.

**Prevention:** Enforce a hard limit on the number of output features (e.g., `MAX_OUTPUT_FEATURES = 100_000`) for all combinatorial functions. Verify this limit using `math.comb` or similar formulas at the start of the function.
