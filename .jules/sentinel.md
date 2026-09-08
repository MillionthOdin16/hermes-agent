
## 2026-05-18 - Sanitize environment to prevent credential leakage in shell execution
**Vulnerability:** Unsanitized environment variables (including API keys) passed to `subprocess.run` in `shell.exec` handling inside `tui_gateway/methods_tools.py`.
**Learning:** The `shell.exec` method in the TUI server process receives all OS environment variables by default. When spawning child processes with `subprocess.run`, it can easily leak sensitive API keys to the child processes or system environment if not explicitly sanitized.
**Prevention:** Use `build_subprocess_env` from `tools.environments.local` and pass `env=sanitized_env` to `subprocess.run` to ensure sensitive credentials are stripped from the environment before executing shell commands.
