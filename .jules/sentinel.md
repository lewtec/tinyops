
## 2026-01-17 - Prevent DoS in Polynomial Features

**Vulnerability:** The `polynomial_features` function lacked input validation for the resulting feature space size. A malicious or accidental input with high `degree` and `n_features` (e.g., 50 features, degree 8) could trigger a combinatorial explosion (1.9 billion features), leading to memory exhaustion and Denial of Service.

**Learning:** Combinatorial algorithms must always validate the magnitude of their output before execution, especially in shared environments. `itertools` makes it easy to create massive iterators lazily, but consuming them or their consequences (like `Tensor.cat` on the result) is dangerous.

**Prevention:** Added a pre-computation check using `math.comb` to verify the total number of output features does not exceed a safety limit (100,000) before proceeding with generation.
