## 2026-02-23 - Enforce usedforsecurity=False in non-cryptographic hashes
**Vulnerability:** Use of weak MD5/SHA1 hashes triggered Bandit B324 high severity warnings.
**Learning:** `hashlib.md5()` and `hashlib.sha1()` were used for non-cryptographic purposes (e.g., generating cache keys and dedup fingerprints). This triggered security scanner false positives and may cause failures in FIPS-compliant environments.
**Prevention:** Always append the `usedforsecurity=False` flag when calling `hashlib.md5()` or `hashlib.sha1()` for non-cryptographic purposes.
