## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-02-05 - DoS in Polynomial Features

**Vulnerability:** The `polynomial_features` function in `tinyops/ml` lacked input validation, allowing users to generate an exponentially large number of features (e.g., `degree=10` with `n_features=100`) leading to memory exhaustion and Denial of Service (DoS).

**Learning:** Combinatorial functions must always validate the expected output size before execution. Using `math.comb` allows for O(1) pre-calculation of result size to fail fast.

**Prevention:** Added a `MAX_OUTPUT_FEATURES` limit of 1,000,000 and implemented a check using `math.comb` to raise `ValueError` if the limit is exceeded.
