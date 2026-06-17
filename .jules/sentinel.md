## 2026-06-16 - [Fix Weak Hash Algorithms]
**Vulnerability:** Used `hashlib.md5()` and `hashlib.sha1()` without the `usedforsecurity=False` parameter in non-cryptographic contexts like cache keys, deduping, and file chunking.
**Learning:** These algorithms are considered weak and can cause tools like Bandit (Rule B324) to flag them as security risks. In certain constrained environments (e.g., FIPS-compliant systems), executing these functions without explicitly marking them as non-security-related will throw errors and cause application crashes.
**Prevention:** Always append the `usedforsecurity=False` flag when utilizing `hashlib.md5()` or `hashlib.sha1()` for non-cryptographic purposes (e.g., caching, chunking, deduplication).
