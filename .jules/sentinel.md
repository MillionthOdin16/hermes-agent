## 2026-10-09 - [Fix Credential Leakage in quick commands]
**Vulnerability:** The quick command exec feature used subprocess.run directly instead of the secure wrapper _captured_exec, which handles errors and timeout securely.
**Learning:** Raw subprocess.run without _captured_exec wrapper bypassing secure timeout handling and standard error capture is an issue. Using _captured_exec mitigates Credential Leakage by ensuring standard output redaction even on errors or timeouts and providing robust timeout management.
**Prevention:** Always use _captured_exec instead of raw subprocess.run for handling TUI RPC shell executions.
