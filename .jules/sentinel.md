## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-20 - Unbounded Combinatorial Explosion in `polynomial_features`

**Vulnerability:** The `polynomial_features` function generated combinations of features without validating the resulting size. For large inputs (e.g., 100 features, degree 10), this would result in trillions of features (42 trillion in the test case), causing a Denial of Service (DoS) via CPU hanging and potential memory exhaustion.

**Learning:** Combinatorial algorithms (using `itertools`) can easily explode in size. Even if the code looks simple, the mathematical properties of combinations mean that small input changes can lead to massive output sizes. This is a classic "algorithmic complexity" vulnerability.

**Prevention:** Always calculate the expected size of combinatorial outputs using `math.comb` *before* starting the generation or allocation loop. Define a safe upper limit (e.g., `MAX_OUTPUT_FEATURES = 100_000`) and raise a `ValueError` if exceeded.
