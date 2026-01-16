# Consistently Ignored Changes

This file lists patterns of changes that have been consistently rejected by human reviewers. All agents MUST consult this file before proposing a new change. If a planned change matches any pattern described below, it MUST be abandoned.

---

## IGNORE: Do not add input validation to the WAV encoder

**- Pattern:** Do not add input validation logic to the `tinyops/io/encode_wav.py` file. While the WAV *decoder* has validation for security reasons, similar changes to the *encoder* are consistently rejected.
**- Justification:** Multiple pull requests attempting to add symmetric validation to the WAV encoder have been closed. This indicates that the change, while potentially well-intentioned, is considered unwanted noise or unnecessary for the project's goals.
**- Files Affected:** `tinyops/io/encode_wav.py`

---

## IGNORE: Do not attempt to fix DoS vulnerabilities in the WAV decoder

**- Pattern:** Do not add validation to the `tinyops/io/decode_wav.py` file to address Denial of Service (DoS) vulnerabilities.
**- Justification:** A pull request (e.g., #142) attempting to fix a DoS vulnerability by adding header validation to the WAV decoder was rejected. This indicates a consistent pattern of rejecting any changes, including security fixes, to the WAV I/O modules.
**- Files Affected:** `tinyops/io/decode_wav.py`
