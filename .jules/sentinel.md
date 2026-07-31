## 2026-05-20 - SSRF and Arbitrary File Read via urllib.request.urlopen
**Vulnerability:** `urllib.request.urlopen` was used without scheme validation in `tui_gateway/server.py` (`_http_ok`), allowing arbitrary local file reads via the `file://` scheme and SSRF.
**Learning:** Python's `urllib.request.urlopen` transparently supports `file://` and `ftp://` schemes by default, which can lead to local file disclosure or SSRF if user-supplied or dynamically generated URLs are not strictly validated.
**Prevention:** Always explicitly validate that the URL scheme starts with `http://` or `https://` before passing it to `urllib.request.urlopen`. Append `# nosec B310` to suppress Bandit warnings once the validation is in place.
