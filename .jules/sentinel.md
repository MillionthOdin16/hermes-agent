## 2026-05-18 - [Add usedforsecurity=False flag to non-cryptographic hashes]
**Vulnerability:** Weak hashing algorithms (MD5, SHA-1) were used for non-cryptographic purposes (caching, chunking) without the `usedforsecurity=False` flag.
**Learning:** In FIPS-compliant environments, cryptographically weak algorithms like MD5 and SHA-1 are blocked at the OpenSSL level, causing `hashlib.md5()` to throw an exception and crash the application.
**Prevention:** When using `hashlib.md5()` or `hashlib.sha1()` for non-cryptographic purposes (e.g., caching, file chunking), always include the `usedforsecurity=False` flag to comply with Bandit security guidelines and FIPS requirements.
