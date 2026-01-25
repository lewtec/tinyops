## 2024-07-25 - User Override on `curl | sh` Installation

**Vulnerability:** Initially identified the `curl | sh` installation method for the `mise` tool in `AGENTS.md` as a critical remote code execution (RCE) vulnerability.

**Learning:** The user explicitly stated that this installation method is **intentional and considered safe** for the project's specific agent environment. This indicates that project-specific context can override general security best practices. The environment is assumed to have controls that mitigate this risk.

**Prevention:** Before flagging common vulnerabilities, consider that the project might have specific, unstated environmental contexts. However, continue to flag them and allow the user to make the final determination. Always document such decisions in the journal to retain context for future sessions.

## 2026-01-25 - [DoS Prevention in CountVectorizer]

**Vulnerability:** The `count_vectorizer` function (and by extension `tfidf_vectorizer`) was vulnerable to Denial of Service (DoS) via memory exhaustion. It built a vocabulary and output matrix proportional to the number of unique tokens in the input corpus without any limit. A malicious input with many unique tokens could cause an Out-Of-Memory (OOM) crash.

**Learning:** Data processing functions that aggregate unique values from potentially unbounded inputs (like text tokenization) must enforce upper bounds on the size of generated structures.

**Prevention:** Implement limits (e.g., `max_features`) on the size of vocabularies or feature sets. Use standard patterns from established libraries (like scikit-learn's `max_features`) to allow users to control resource usage.
