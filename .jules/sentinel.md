## 2026-03-05 - Fix SSRF in tui_gateway via URL scheme validation
**Vulnerability:** The `_http_ok` function in `tui_gateway/server.py` uses `urllib.request.urlopen` with user-supplied URLs (via CDP connect endpoints) without validating that the URL scheme is `http` or `https`. This could allow Server-Side Request Forgery (SSRF) and arbitrary local file reads via the `file://` scheme.
**Learning:** `urllib.request.urlopen` automatically handles `file://` URIs by default, which can lead to local file disclosure if user-controlled input or compromised external endpoints are passed as URLs.
**Prevention:** Always explicitly validate the URL scheme before calling `urllib.request.urlopen` and append `# nosec B310` to suppress Bandit warnings once validated.
