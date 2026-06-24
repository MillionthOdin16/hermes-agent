## 2026-06-24 - SSRF Prevention in URL Probing
**Vulnerability:** Potential SSRF/Path Traversal via arbitrary URL schemes (like `file://`) allowed by `urllib.request.urlopen`.
**Learning:** Python's `urlopen` natively supports `file://` which can accidentally expose local files if the URL input is unconstrained.
**Prevention:** Always explicitly validate that user-provided or unconstrained URLs start with permitted schemes (`http://` or `https://`) before passing them to `urlopen`, and apply `# nosec B310` suppression once validated.