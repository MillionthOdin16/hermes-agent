## 2026-07-14 - Fix Server-Side Request Forgery (SSRF)

**Vulnerability:** Found missing scheme validation on `urllib.request.urlopen` in `tui_gateway/server.py` which could lead to SSRF or arbitrary local file reads via the `file://` scheme.
**Learning:** Python's `urllib.request.urlopen` supports multiple schemes (including `file://` by default) which can expose local files if the URL isn't explicitly validated.
**Prevention:** Always validate URL schemes (e.g., `startswith(("http://", "https://"))`) before passing them to `urlopen` and add `# nosec B310` to indicate it was handled.
