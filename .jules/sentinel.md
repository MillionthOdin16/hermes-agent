## 2026-07-18 - Fix SSRF and local file read vulnerability in OSV check
**Vulnerability:** urllib.request.urlopen was called with _OSV_ENDPOINT (configurable via environment variable) without scheme validation, allowing file:// scheme usage.
**Learning:** Configurable endpoints used in urllib.request.urlopen can lead to SSRF or arbitrary local file reads if the scheme is not explicitly validated.
**Prevention:** Always explicitly validate the URL scheme (e.g., ensuring it starts with http:// or https://) before passing it to urllib.request.urlopen, and use # nosec B310 to suppress Bandit warnings once validated.
