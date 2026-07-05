## 2026-07-05 - Prevent SSRF in file downloads
**Vulnerability:** Unrestricted URL schemas passed to `urllib.request.urlopen` allowing Server-Side Request Forgery (SSRF) and local arbitrary file read via `file://`.
**Learning:** Python's urllib fetches file paths natively unless strictly validated for web schemes.
**Prevention:** Always validate `url.startswith(("http://", "https://"))` before passing it to `urllib.request.urlopen` and suppress the Bandit warning only after explicit validation.
