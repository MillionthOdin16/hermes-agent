## 2026-07-16 - SSRF and Local File Read via urllib.request.urlopen
**Vulnerability:** Found unvalidated URL inputs being passed directly to `urllib.request.urlopen` in `tui_gateway/server.py` (`_http_ok` function).
**Learning:** Python's `urllib.request.urlopen` supports multiple schemes, including `file://`. Without explicit scheme validation, an attacker could supply a `file://` URL to read arbitrary local files or an internal network URL to perform Server-Side Request Forgery (SSRF).
**Prevention:** Always explicitly validate the URL scheme (e.g., ensuring it starts with `http://` or `https://`) before passing it to `urlopen`.
