## 2026-09-18 - Sanitize environment for shell.exec to prevent credential leakage
**Vulnerability:** The `shell.exec` command in `tui_gateway/methods_tools.py` was executing `subprocess.run` with `shell=True` without sanitizing the environment variables, which could leak sensitive credentials stored in `os.environ`.
**Learning:** Even if commands are gated or deemed "safe", running them in an environment that contains all API keys and secrets is dangerous. The TUI server process runs with elevated secrets.
**Prevention:** Always use `tools.environments.local.build_subprocess_env()` to construct a sanitized environment before executing subprocesses that could potentially execute untrusted or shell commands, especially when `shell=True` is used.
