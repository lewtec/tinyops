## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-17 - DoS Protection in Polynomial Features

**Vulnerability:** The `polynomial_features` function lacked input validation for the `degree` parameter. An attacker could supply a large `degree` (e.g., 50), causing a combinatorial explosion in feature generation. This would lead to excessive memory consumption and CPU usage, resulting in a Denial of Service (DoS).

**Learning:** Functions that perform combinatorial expansions must validate input parameters that control the expansion size. In `tinyops`, which aims for graph compilation, large expansions can also cause compilation timeouts or graph size explosions.

**Prevention:** Always implement reasonable upper bounds on input parameters that drive combinatorial logic (e.g., `degree`, `n_frames`). These limits should be based on practical use cases and system resource constraints.
