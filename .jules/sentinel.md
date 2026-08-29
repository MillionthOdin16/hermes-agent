## 2026-08-29 - Prevent credential leakage in `shell.exec` subprocess
**Vulnerability:** Subprocesses spawned by `shell.exec` inherited the full environment containing sensitive API keys from the TUI server process.
**Learning:** In environments like `tui_gateway` where `os.environ` holds secrets, passing a sanitized environment to `subprocess.run` is critical to prevent leakage to user-provided commands.
**Prevention:** Always explicitly pass a sanitized environment (e.g. `env=build_subprocess_env()`) to `subprocess.run` to limit exposure.
