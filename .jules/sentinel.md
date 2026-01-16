## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2024-07-26 - DoS Vulnerability in `polynomial_features`

**Vulnerability:** The `polynomial_features` function in `tinyops/ml/polynomial_features.py` lacked input validation for the `degree` parameter. A large `degree` value could lead to a combinatorial explosion in the number of features, causing excessive memory allocation and leading to a Denial of Service (DoS).

**Learning:** This vulnerability highlights a common pattern where the lack of input validation on parameters that control resource allocation can lead to DoS vulnerabilities. A simple check for negative values is insufficient if the primary risk comes from large positive values. A complete fix requires enforcing a reasonable upper bound.

**Prevention:** Always validate input parameters that control the size or complexity of the output. This includes checking for both nonsensical values (e.g., negative numbers) and values that, while technically valid, could lead to resource exhaustion. Enforcing a reasonable upper limit is a critical defense against DoS attacks in such functions.
