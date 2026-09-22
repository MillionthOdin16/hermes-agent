
## 2026-06-14 - Nested Icon-Only Button ARIA Labels
**Learning:** When an icon-only interactive element (like `<Button size="icon">`) is nested inside a wrapper link (`<a>` tag) that provides a `title` or `aria-label`, the innermost interactive element still needs its own `aria-label` attribute. Otherwise, screen readers may not correctly identify the interactive button itself.
**Action:** Always verify that innermost interactive elements have their own `aria-label`, even if a parent wrapper already provides accessible context.
