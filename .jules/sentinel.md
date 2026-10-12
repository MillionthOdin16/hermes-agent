## 2026-06-12 - Prevent Credential Leakage via child process environment injection

**Vulnerability:** Found multiple subprocess.run calls with `shell=True` executing user or project-authored shell commands without sanitizing the environment variables. Because the Hermes process holds sensitive credentials (like API keys) in `os.environ`, any child process spawned via `shell=True` would inherit these credentials. If an attacker controls the command string (or even just part of it), they could exfiltrate these sensitive variables to an external server.

**Learning:** This is a critical security issue regarding credential leakage rather than command injection directly, as sanitizing the environment doesn't prevent shell injection itself, but mitigates the risk of exfiltrating sensitive environment variables to an untrusted source. I must use `tools.environments.local.build_subprocess_env()` to strip out secrets before spawning child processes.

**Prevention:** Always use `build_subprocess_env()` (or `served_profile_child_env` / `delegated_child_subprocess_env`) to build a sanitized environment for `subprocess.run` (and `subprocess.Popen`) calls that execute external shell scripts or user-provided commands. Never pass the raw `os.environ` to untrusted shell executions.
