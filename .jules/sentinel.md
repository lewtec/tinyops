## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2024-07-26 - Algorithmic Complexity DoS

**Vulnerability:** Functions that perform computationally expensive operations (e.g., pairwise distance calculations in `nearest_neighbors`) without validating the size of the input data are susceptible to Denial of Service (DoS) attacks. A malicious user could provide a very large input, causing the system to exhaust CPU resources.

**Learning:** The codebase has a recurring pattern of missing input size validation, especially in the `ml` module. This is a critical vulnerability that needs to be addressed proactively.

**Prevention:** When implementing or reviewing functions that have non-linear algorithmic complexity, always add input validation to enforce a reasonable upper bound on the input size. This mitigates the risk of resource exhaustion.
