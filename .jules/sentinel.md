
## 2026-03-01 - [Fix weak hashing warnings in caching]
**Vulnerability:** Bandit B324 raised for `hashlib.md5()` used in non-cryptographic contexts like cache keys and file hashing.
**Learning:** Python 3.9+ supports `usedforsecurity=False` in `hashlib` to explicitly denote non-cryptographic usage, which silences false positive security alerts and avoids crashing on FIPS-compliant systems.
**Prevention:** Always include `usedforsecurity=False` when using md5/sha1 for hashing non-sensitive data like cache keys or file contents.
