## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-10 - Symmetric Input Validation for WAV Encoding

**Vulnerability:** The `decode_wav` function included a `MAX_WAV_FRAMES` check to prevent a Denial of Service (DoS) attack from maliciously crafted WAV files with huge headers. However, the `encode_wav` function lacked a corresponding check, meaning the application could create oversized files that its own decoder would refuse to read.

**Learning:** Security controls must be applied symmetrically. If a decoder has input validation, the corresponding encoder must have parallel checks to prevent the creation of invalid or dangerous outputs. This ensures internal consistency and prevents self-inflicted DoS vulnerabilities.

**Prevention:** When implementing paired functions (e.g., encoders/decoders, serializers/deserializers), always ensure that security validation logic is mirrored between them. A change in one should trigger a review of the other.
