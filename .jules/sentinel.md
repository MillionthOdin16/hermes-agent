## 2026-06-23 - [Suppressing False Positive Command Injection in Arbitrary Shell Endpoints]
**Vulnerability:** [Bandit flagged a subprocess.run with shell=True in tui_gateway/server.py.]
**Learning:** [Endpoints explicitly designed for arbitrary shell execution (like shell.exec and quick commands) strictly require shell=True to support shell features like pipes and redirects. Converting them to shell=False breaks core functionality and crashes on empty inputs.]
**Prevention:** [Suppress Bandit warnings with # nosec B602 only when external safety validations (like detect_dangerous_command) are in place to validate the input before execution.]
