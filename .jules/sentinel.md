## 2026-03-24 - Securing shell.exec subprocess environments in tui_gateway
**Vulnerability:** Command injection risk in `shell.exec` due to potential credential leakage in subprocesses.
**Learning:** In the `tui_gateway` environment, `os.environ` can contain sensitive API keys. When spawning subprocesses (especially with `shell=True`), this environment is inherited by default, potentially exposing credentials to untrusted commands. Using `shell=True` is the intended behavior for `shell.exec`, so we must instead sanitize the environment.
**Prevention:** Always explicitly pass a sanitized environment to `subprocess.run` by importing `build_subprocess_env` from `tools.environments.local` and passing `env=build_subprocess_env()` in `tui_gateway/methods_tools.py` and similar locations.
