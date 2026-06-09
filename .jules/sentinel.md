
## 2026-06-09 - Ensure non-cryptographic MD5 usage explicitly passes Bandit audits
**Vulnerability:** Weak cryptographic hashes (`hashlib.md5`) were being used without explicit intent flags, leading to high-severity `bandit` security lint warnings across the codebase.
**Learning:** `hashlib.md5()` flags a security warning by default because MD5 is cryptographically broken. When using MD5 for non-cryptographic purposes (like cache keys or diff checking), it triggers false positives.
**Prevention:** Always append `usedforsecurity=False` when calling `hashlib.md5()` (or `hashlib.sha1()`) if the hash is not used in a cryptographic context. This signals intent and ensures the code passes `bandit` security checks cleanly.
