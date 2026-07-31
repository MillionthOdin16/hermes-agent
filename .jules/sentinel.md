## 2026-05-18 - [Fix Bandit B324 Weak MD5 warnings]
**Vulnerability:** Weak MD5 hash warnings triggered by `hashlib.md5()`.
**Learning:** `hashlib.md5()` was being used for non-cryptographic purposes (caching, change detection) without explicitly indicating it, which caused Bandit to flag it as a potential security risk (CWE-327).
**Prevention:** Always use `hashlib.md5(usedforsecurity=False)` or `hashlib.sha1(usedforsecurity=False)` when hashing for non-security reasons such as caching, indexing, or chunking.
