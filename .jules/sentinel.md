## 2026-06-25 - Fix SSRF Vulnerability in URL Probing
**Vulnerability:** The `_http_ok` function used `urllib.request.urlopen` on unvalidated URLs, creating a Server-Side Request Forgery (SSRF) and local file read vulnerability (e.g., via `file://`).
**Learning:** Built-in Python URL fetchers do not restrict schemes by default. An unvalidated URL input can expose the local file system or internal network services.
**Prevention:** Always validate URL schemes (e.g., ensuring they start with `http://` or `https://`) before passing them to HTTP client libraries.
