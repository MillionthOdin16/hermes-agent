## 2026-10-15 - Specify usedforsecurity=False in hashlib.md5
**Vulnerability:** Use of weak MD5 hash without `usedforsecurity=False` flag.
**Learning:** Using `hashlib.md5()` without explicitly marking `usedforsecurity=False` when the hash is used for non-security contexts (like caching, e.g., in `tools/skills_hub.py`) triggers Bandit warnings and can be a concern on platforms in FIPS mode.
**Prevention:** Always append `usedforsecurity=False` in `hashlib.md5(..., usedforsecurity=False)` or `hashlib.sha1(...)` when computing hashes purely for indexing, caching, or file uniqueness checking.
