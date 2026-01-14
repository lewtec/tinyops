## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2024-07-26 - DoS Vulnerability in WAV Decoder

**Vulnerability:** The `decode_wav` function in `tinyops/io/decode_wav.py` was missing input validation for the number of channels (`n_channels`) and the sample rate (`framerate`) specified in a WAV file's header. A malicious actor could craft a WAV file with an extremely large number of channels, bypassing the existing `MAX_WAV_FRAMES` check and triggering a massive memory allocation, leading to a Denial of Service (DoS).

**Learning:** Security checks often focus on the most obvious parameters (like the number of frames), but secondary parameters (like channel count) can be just as dangerous if they are a factor in memory allocation calculations. All externally controlled inputs that influence resource allocation must be validated.

**Prevention:** When parsing complex file formats, create a threat model that considers every header field. Implement strict, reasonable limits for all parameters that influence memory allocation or processing time, not just the primary ones.
