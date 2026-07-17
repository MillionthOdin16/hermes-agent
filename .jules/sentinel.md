## 2026-10-27 - URL Scheme Validation for urlopen
**Vulnerability:** Arbitrary local file read / SSRF via `file://` scheme in `urllib.request.urlopen`.
**Learning:** Functions accepting URLs to download artifacts or connect to services can be exploited if they don't validate the URL scheme, allowing attackers to access local files.
**Prevention:** Explicitly check that the URL starts with `http://` or `https://` before calling `urllib.request.urlopen`, and suppress the Bandit warning with `# nosec B310` once validated.
