## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-16 - DoS via Malformed WAV Header Metadata

**Vulnerability:** The `decode_wav` function was vulnerable to a Denial of Service (DoS) attack. While it validated the number of frames (`n_frames`), it failed to validate the channel count (`n_channels`) and sample rate (`framerate`). A malicious WAV file with extremely large values for these fields could trigger excessive memory allocation, causing the application to crash.

**Learning:** This is a classic example of incomplete input validation in a file parser. The initial fix for `n_frames` was a good start, but it's crucial to validate *all* header fields that influence resource allocation. A single unchecked parameter can be an attack vector. The principle of "Trust nothing, verify everything" applies to every piece of metadata in a file header.

**Prevention:** When implementing or reviewing file parsing logic, treat the entire header as untrusted input. Create a threat model for each field: How could this value be abused if it were maliciously large or small? Implement strict, reasonable limits for every parameter that controls memory allocation, loop iterations, or other resource-intensive operations.
