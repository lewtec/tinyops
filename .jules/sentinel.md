## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-20 - DoS Protection in Polynomial Features

**Vulnerability:** `polynomial_features` lacked input size validation, allowing combinatorial explosion of output features (e.g., millions of columns) via the `degree` parameter, leading to memory exhaustion and Denial of Service.

**Learning:** Python's `itertools` and loop-based logic for combinatorial generation can easily become a bottleneck or crash vector if not guarded against large inputs.

**Prevention:** Always pre-calculate the expected size of combinatorial outputs using `math.comb` or similar efficient methods and validate against a hard limit before starting generation or allocation.
