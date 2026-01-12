## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.
## 2026-01-12 - Add Symmetric Validation to WAV Encoder

**Vulnerability:** The  function included a security check to limit the number of frames in a WAV file header () to prevent a Denial of Service (DoS) attack via excessive memory allocation. However, the  function lacked a corresponding check, allowing a user to create an excessively large WAV file by passing in a massive tensor, leading to the same DoS vulnerability.

**Learning:** Security controls must be applied symmetrically. If an input channel (decoder) has validation to protect against a specific threat, the corresponding output channel (encoder) must have a parallel check to prevent the *creation* of a malicious payload of the same type. Relying on validation at only one end of the I/O pipeline leaves a gap.

**Prevention:** When implementing or reviewing features with corresponding input and output functions (e.g., encoders/decoders, parsers/serializers), always verify that any security validation applied to one is mirrored in the other.
