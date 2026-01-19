## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-19 - DoS Vulnerability in Polynomial Features

**Vulnerability:** The `polynomial_features` function in `tinyops/ml/polynomial_features.py` lacked input validation for the resulting feature space size. A malicious or accidental input with a large `degree` or `n_features` could cause the function to attempt allocating an extremely large amount of memory, leading to a Denial of Service (DoS) via memory exhaustion.

**Learning:** Combinatorial functions are high-risk for DoS. The `math.comb` function is an efficient way to pre-calculate the expected size of such operations without actually generating the combinations, allowing for safe early rejection of dangerous inputs.

**Prevention:** Always validate the output size of combinatorial or expansive operations against a safe hard limit (e.g., `MAX_OUTPUT_FEATURES`) before beginning execution or memory allocation. Use lightweight mathematical formulas to predict resource usage.
