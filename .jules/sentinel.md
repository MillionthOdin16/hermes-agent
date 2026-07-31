## 2026-06-15 - Validate URL Schemes for SSRF Prevention
**Vulnerability:** Arbitrary local file reads via the 'file://' scheme when using `urllib.request.urlopen` in `tui_gateway/server.py`.
**Learning:** `urllib.request.urlopen` supports local file paths by default unless restricted, leading to potential SSRF and local file exposure.
**Prevention:** Always validate that URLs begin with `http://` or `https://` before passing them to `urlopen`. Suppress Bandit false positives with `# nosec B310` once validated.
