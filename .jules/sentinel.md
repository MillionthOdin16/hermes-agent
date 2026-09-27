## 2026-06-12 - Secure quick commands execution against credential leakage
**Vulnerability:** Quick commands (`_dispatch_quick` for exec types) leaked sensitive output across the RPC boundary without force redaction or URL credential redaction, leaving credentials vulnerable in transcripts.
**Learning:** `shell.exec` correctly used `force=True` and `redact_url_credentials=True` when redacting, but the quick command execution path missed this robust redaction configuration.
**Prevention:** Always use identical and complete redaction configurations (`force=True, redact_url_credentials=True`) across all `shell.exec` and equivalent subprocess RPC boundaries.
