## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2024-07-26 - Symmetrical I/O Validation for WAV Encoding

**Vulnerability:** The `decode_wav` function was previously hardened with a `MAX_WAV_FRAMES` limit to prevent Denial of Service (DoS) attacks from malformed WAV files. However, the corresponding `encode_wav` function lacked a similar check, creating an asymmetric security control. An attacker could craft a large tensor to pass to the encoder, triggering a DoS by forcing a massive memory allocation.

**Learning:** Security controls on data parsing or processing functions must be applied symmetrically. If a decoder has input validation, the corresponding encoder must have equivalent validation to prevent the creation of the very data that is considered insecure to decode. This is a common pattern in I/O libraries.

**Prevention:** When reviewing a file handler or data processor, always check for its counterpart (e.g., encoder for a decoder, writer for a reader). Ensure that security validation logic is shared or mirrored between them to prevent asymmetric vulnerabilities.
