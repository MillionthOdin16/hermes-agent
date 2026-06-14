## 2026-05-18 - [Fix Insecure Subprocess Shell=True]
**Vulnerability:** Use of shell=True in subprocess calls which can lead to command injection if inputs are not properly sanitized.
**Learning:** Using shell=True executes the command through the system shell, which is unnecessary and unsafe when arguments can be passed securely as a list.
**Prevention:** Always use a list of arguments for subprocess commands and avoid shell=True.
