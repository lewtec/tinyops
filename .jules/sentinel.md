## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-13 - Symmetric Validation in I/O Operations

**Vulnerability:** The `encode_wav` function lacked input validation on the number of audio frames. A malicious actor could pass a tensor with an extremely large number of frames, triggering a massive memory allocation that would lead to a Denial of Service (DoS) and crash the application.

**Learning:** This vulnerability highlighted an asymmetry in the I/O logic. While the `decode_wav` function correctly implemented a security limit (`MAX_WAV_FRAMES`) to protect against malformed files, the `encode_wav` function did not have a corresponding check. This meant the application could be used to create a malicious file that its own decoder would reject, a classic example of an application layer DoS vulnerability.

**Prevention:** Paired I/O functions (e.g., encoders/decoders, serializers/deserializers) must have symmetric validation logic. Security controls applied to data consumption (reading, decoding) must be mirrored in data production (writing, encoding) to prevent the application from creating unsafe or malicious outputs. Reusing the same validation constants (like `MAX_WAV_FRAMES`) across both functions is a good practice to ensure consistency.
