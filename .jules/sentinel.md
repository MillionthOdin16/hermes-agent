## 2026-03-01 - SSRF in tui_gateway/server.py
**Vulnerability:** The `_http_ok` function calls `urllib.request.urlopen` with user-supplied URLs without verifying the URL scheme, which can lead to Server-Side Request Forgery (SSRF) and allow local file reads (e.g. `file:///`).
**Learning:** `urlopen` blindly follows arbitrary schemes unless restricted. This allows reading arbitrary local files or making requests to internal endpoints.
**Prevention:** Always validate URL schemes before passing them to `urllib.request.urlopen` (e.g., `url.lower().startswith(('http://', 'https://'))`), and suppress the subsequent Ruff warning with `# noqa: S310`.
