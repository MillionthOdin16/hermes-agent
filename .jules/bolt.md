## 2026-05-18 - Avoid defining inner functions inside frequently called outer functions
**Learning:** In Python, defining inner functions (closures) inside outer functions adds runtime execution overhead as the function is recreated on every invocation of the outer function. This becomes a performance bottleneck if the outer function is on a hot path, like `_extract_error_code` which might be called for every error.
**Action:** Extract the inner function `_code_from_payload` out of `_extract_error_code` to module-level scope. Update `agent/error_classifier.py` and submit PR.
