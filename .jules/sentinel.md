## 2026-05-13 - [Sentinel] SSRF in `urllib.request.urlopen`
**Vulnerability:** URL requests using `urllib.request.urlopen` were made without explicitly validating the URL scheme, which allows Server-Side Request Forgery (SSRF) and local file reads (via `file://`).
**Learning:** Using `urllib.request.urlopen` without schema validation is risky because it natively supports potentially dangerous schemes like `file://` or `ftp://` which can be exploited if input is user-controlled.
**Prevention:** Always validate that the URL begins with `http://` or `https://` before passing it to `urllib.request.urlopen`, and suppress the Bandit warning with `# nosec B310` once safely validated.
