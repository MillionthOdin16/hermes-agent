## 2026-09-30 - Environment Sanitization for subprocess with shell=True
**Vulnerability:** Credential Leakage / Command Injection
**Learning:** `subprocess.run(..., shell=True)` without explicit environment sanitation can leak the long-lived parent process's environment variables (including sensitive API keys and secrets) to the child process.
**Prevention:** Always secure `subprocess.run` calls that require `shell=True` by passing `env=build_subprocess_env()` to sanitize the environment.
