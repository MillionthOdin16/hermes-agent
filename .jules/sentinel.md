## 2026-09-02 - Environment leakage in subprocess call
**Vulnerability:** In `tui_gateway/methods_tools.py`, the `shell.exec` method invoked `subprocess.run` with `shell=True` but failed to sanitize the environment (`os.environ`). Because this runs in the TUI server process, it leaks all API keys into the child process environment.
**Learning:** Even if a method runs arbitrary commands on purpose, the environment must always be sanitized before calling `subprocess.run` (or similar execution functions) to avoid inadvertently exposing credentials to the child process.
**Prevention:** Always use `build_subprocess_env` from `tools.environments.local` to explicitly sanitize the environment passed to `subprocess.run` via the `env=` argument.
