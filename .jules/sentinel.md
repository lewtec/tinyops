## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-16 - DoS Prevention in Polynomial Features

**Vulnerability:** The `polynomial_features` function in `tinyops/ml/polynomial_features.py` lacked input validation, allowing users to request a combinatorially explosive number of output features (e.g., millions). This would lead to memory exhaustion (DoS) as the system attempted to allocate massive tensors.

**Learning:** Combinatorial algorithms must always have safety rails. Even if the underlying library (tinygrad) or the reference implementation (sklearn) handles it by crashing or attempting to run, a security-focused library should preemptively block obviously dangerous inputs.

**Prevention:** I implemented a pre-calculation check using `math.comb` to predict the number of output features. If the count exceeds a safe threshold (100,000), the function now raises a `ValueError` before allocating any memory. A regression test `tinyops/ml/_dos_test.py` was added to enforce this limit.
