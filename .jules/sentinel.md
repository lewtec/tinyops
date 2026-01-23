## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-17 - DoS Prevention in Polynomial Features

**Vulnerability:** The `polynomial_features` function lacked input validation for the `degree` parameter. A negative degree caused logical inconsistencies (returning bias or empty), while an excessively large degree (relative to the number of features) could trigger a combinatorial explosion, leading to Denial of Service (DoS) via memory exhaustion.

**Learning:** Mathematical or combinatorial functions often have "hidden" complexity. Just because the code looks simple (a loop) doesn't mean it's safe. Specifically, `itertools.combinations` grows factorially/polynomially. We must validate parameters that control output size against a reasonable upper bound to prevent resource exhaustion attacks.

**Prevention:** When reviewing functions that generate combinations, permutations, or large grids (like `meshgrid` or `polynomial_features`), always check if the output size is bounded. If not, implement a hard limit on the expected number of elements before starting the computation.
