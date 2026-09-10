## 2026-05-18 - Prevent Credential Leakage in shell.exec
**Vulnerability:** The `shell.exec` method in `tui_gateway/methods_tools.py` spawns subprocesses without sanitizing the environment. Since it runs in the TUI server process, all API keys from `os.environ` are passed down to arbitrary user-controlled shell commands, creating a critical credential leakage risk.
**Learning:** Even if a command is approved (e.g., via `detect_dangerous_command`), it can still access the parent's environment variables.
**Prevention:** Always explicitly pass a sanitized environment (e.g., `env=build_subprocess_env()`) to `subprocess.run` inside `tui_gateway` and similar sensitive contexts where `os.environ` contains secrets.
