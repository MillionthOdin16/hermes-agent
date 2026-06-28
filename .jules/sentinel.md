## 2026-05-24 - [CRITICAL] Prevent SSRF and File Exfiltration in `tui_gateway/server.py`
**Vulnerability:** The `_http_ok` function used `urllib.request.urlopen(url)` without validating the scheme, allowing requests to arbitrary protocols, including `file://`.
**Learning:** This could be exploited for Server-Side Request Forgery (SSRF) or arbitrary local file reads, as Bandit correctly flagged (B310).
**Prevention:** Always validate URL schemes (e.g., restrict to `http` and `https`) before passing them to `urlopen`. Suppress Bandit warnings with `# nosec B310` only after robust scheme validation is in place.
