## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-16 - DoS Prevention in Polynomial Features

**Vulnerability:** The `polynomial_features` function in `tinyops/ml` lacked input validation, allowing users to request a degree and feature count that would result in combinatorial explosion (e.g., billions of features), leading to memory exhaustion (DoS).

**Learning:** Mathematical operations that expand the feature space factorially or exponentially (like polynomial combinations) must be validated *before* allocation. Using `math.comb` provides a cheap way to predict the memory cost.

**Prevention:** Always calculate the expected output size of generative functions against a hard limit (e.g., `MAX_OUTPUT_FEATURES`) before beginning execution or allocation.
