## 2026-08-04 - SSRF URL Validation in TUI Gateway
**Vulnerability:** The `_http_ok` function in `tui_gateway/server.py` passed unvalidated user-controlled URLs directly to `urllib.request.urlopen`, creating an SSRF (Server-Side Request Forgery) and local file read vulnerability via the `file://` scheme.
**Learning:** Python's `urllib.request.urlopen` supports multiple schemes (like `file://`, `ftp://`) by default. Passing arbitrary URLs without scheme validation can lead to unintended access to local files or internal network resources.
**Prevention:** Always explicitly validate the URL scheme case-insensitively (e.g., ensuring `url.lower().startswith(('http://', 'https://'))`) before passing it to `urlopen`. Suppress the Ruff security warning with `# noqa: S310` once validated.
