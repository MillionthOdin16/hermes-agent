## 2024-05-30 - MD5 Hardening
**Vulnerability:** Found uses of `hashlib.md5()` without the `usedforsecurity=False` flag, violating security linting guidelines (Bandit [B324]).
**Learning:** `hashlib.md5()` is considered cryptographically weak and shouldn't be used for security-sensitive operations. When used for non-security purposes like caching or deduping, the `usedforsecurity=False` flag must be explicitly set to indicate this intent and suppress false positive security alerts, as per FIPS compliance requirements.
**Prevention:** Use `hashlib.md5(..., usedforsecurity=False)` for non-security hashing (like caching or chunking) or upgrade to a stronger hashing algorithm like SHA-256 for security purposes.
