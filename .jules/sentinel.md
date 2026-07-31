## 2026-05-18 - Prevent SSRF in `urllib.request.urlopen`
**Vulnerability:** Found unvalidated URL inputs being passed to `urllib.request.urlopen` in `tui_gateway/server.py`.
**Learning:** Using `urllib.request.urlopen` without explicitly validating the URL scheme enables SSRF and arbitrary local file reads (e.g. `file://`).
**Prevention:** Always validate that URLs begin with `http://` or `https://` before fetching them with `urllib`, and append `# nosec B310` once the validation is in place.
