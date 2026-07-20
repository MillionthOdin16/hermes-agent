## 2026-07-20 - Prevent SSRF via URL Scheme Validation
**Vulnerability:** The urllib.request.urlopen call in _http_ok was vulnerable to Server-Side Request Forgery (SSRF) and arbitrary local file reads because the URL scheme was unvalidated.
**Learning:** Python's urlopen automatically handles multiple schemes, meaning any unvalidated URL input can expose local files or internal services unexpectedly.
**Prevention:** Always validate URL input case-insensitively to strictly permit only safe schemes (http://, https://) before passing it to urlopen, then suppress Bandit warnings with # nosec B310.
