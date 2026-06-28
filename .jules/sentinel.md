## 2026-06-03 - [Bandit hashlib.md5 usedforsecurity]
**Vulnerability:** Weak MD5 and SHA1 hash used without `usedforsecurity=False`
**Learning:** Bandit flags `hashlib.md5` and `hashlib.sha1` when `usedforsecurity=False` is missing.
**Prevention:** Include `usedforsecurity=False` when the hash is not for security purposes.
