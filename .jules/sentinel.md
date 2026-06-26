## 2026-06-26 - [MEDIUM] Fix SSRF vulnerability in urlopen
**Vulnerability:** Potential Server-Side Request Forgery (SSRF) and local file read due to unvalidated URL schemes in `urllib.request.urlopen` in `tui_gateway/server.py`.
**Learning:** `urllib.request.urlopen` supports arbitrary schemes like `file://` which can be exploited by an attacker to read local files or reach internal endpoints.
**Prevention:** Always validate that the URL explicitly starts with expected schemes like `http://` or `https://` before passing it to `urlopen`, and use `# nosec B310` to suppress Bandit warnings once properly validated.
