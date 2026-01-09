## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-09 - Missing Input Validation in WAV Encoder

**Vulnerability:** The  function lacked validation on the size of the input tensor. A tensor with an extremely large number of frames could be passed to the function, leading to excessive memory allocation during the conversion to a NumPy array and subsequent byte buffer, causing a Denial of Service (DoS).

**Learning:** Security controls must be applied consistently across related components. The corresponding  function had a  check to prevent DoS from malicious files, but the encoder was missing a parallel check. This oversight created a vulnerability where the application itself could be forced to consume excessive resources.

**Prevention:** When reviewing or implementing related input/output or serialization/deserialization functions, ensure that security validations are symmetric. If a parser has a size limit, the corresponding generator should have one as well.
