## 2026-06-09 - Ensure aria-label on nested icon-only interactive elements
**Learning:** Even when an outer container or link (`<a>`) provides a `title` or `aria-label` attribute (e.g., `title="Open Docs"`), if it wraps an inner icon-only interactive element like a `<Button>`, the inner element may still be incorrectly parsed by screen readers or linters if it lacks its own `aria-label`.
**Action:** Always verify and attach an explicit `aria-label` directly to the innermost icon-only `<Button>` or interactive element, mirroring the parent's descriptive text if necessary.
