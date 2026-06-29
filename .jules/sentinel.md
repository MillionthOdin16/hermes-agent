## 2026-05-18 - SSRF via urllib.request.urlopen
**Vulnerability:** Server-Side Request Forgery (SSRF) and local file read via `urllib.request.urlopen` which allows the `file://` scheme by default.
**Learning:** Python's built-in `urllib.request.urlopen` does not restrict protocols to HTTP/HTTPS, silently enabling unintended local file access if user-supplied or dynamic URLs are passed to it.
**Prevention:** Always validate that URLs explicitly start with `http://` or `https://` before calling `urlopen`. Once validated, append `# nosec B310` to suppress Bandit warnings.
