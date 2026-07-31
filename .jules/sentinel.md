## 2026-06-27 - Fix SSRF Vulnerability in File Downloader
**Vulnerability:** Unvalidated URL scheme used in `urllib.request.urlopen`.
**Learning:** Always explicitly validate URL schemes (e.g., `http://` or `https://`) before passing user-controlled or external URLs to `urllib.request.urlopen` to prevent Server-Side Request Forgery (SSRF) and arbitrary local file reads.
**Prevention:** Enforce strict URL scheme validation and append `# nosec B310` to suppress Bandit warnings only after verification.
