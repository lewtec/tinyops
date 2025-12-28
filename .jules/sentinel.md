## 2024-07-25 - Insecure `curl | sh` Installation Method

**Vulnerability:** The project's `AGENTS.md` documentation recommended installing the `mise` tool using `curl https://mise.run | sh`. This method is dangerous as it executes a remote script without verification, creating a significant supply-chain risk. An attacker compromising the script's server could execute arbitrary code on a developer's machine.

**Learning:** The convenience of a one-line installer was prioritized over security. The project lacked a clear policy on how to handle external dependencies and setup instructions, leading to the adoption of a risky, albeit common, installation pattern.

**Prevention:** Always prefer official, signed package managers (`apt`, `brew`, `dnf`, etc.) for installing external tools. These managers verify package integrity and source, mitigating the risk of remote code execution. The project's documentation must be updated to reflect this security-first approach for all external dependencies.
