## 2026-09-21 - Prevent Credential Leakage in shell.exec
**Vulnerability:** Credential leakage risk in `tui_gateway/methods_tools.py` via `shell.exec`. Since `shell=True` is required for intended functionality, an attacker could inject commands to exfiltrate sensitive environment variables (API keys).
**Learning:** When addressing command injection risks in methods where shell execution is the intended functionality (like `shell.exec`), do not disable `shell=True` or use `shlex.split()`, as this breaks shell operators. Instead, secure the call by sanitizing the environment.
**Prevention:** Use `tools.environments.local.build_subprocess_env()` to sanitize the environment before calling subprocess.run to prevent credential leakage.
