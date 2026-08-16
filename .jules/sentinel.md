## 2026-05-13 - [SSRF and Arbitrary File Read in urllib]
**Vulnerability:** urllib.request.urlopen was called with unsanitized URLs, allowing SSRF and arbitrary local file reads via the 'file://' scheme.
**Learning:** Python's urllib.request.urlopen supports schemes like 'file://' and 'ftp://' by default. If the URL is user-controlled or fetched externally, it poses a critical risk.
**Prevention:** Validate the URL scheme case-insensitively (e.g., ensuring `url.lower().startswith(('http://', 'https://'))`) before passing it to `urlopen`.
