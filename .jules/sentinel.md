## 2026-05-20 - Enforce URL Scheme Validation for urlopen
**Vulnerability:** Server-Side Request Forgery (SSRF) and arbitrary local file reads via unvalidated URLs passed to `urllib.request.urlopen`.
**Learning:** Using `urllib.request.urlopen` without explicitly validating the URL scheme allows the use of the `file://` or custom schemes, which can lead to local file disclosure or unexpected network requests.
**Prevention:** Always validate that the URL starts with `http://` or `https://` before opening it, and append `# nosec B310` to suppress Bandit warnings once validated.
