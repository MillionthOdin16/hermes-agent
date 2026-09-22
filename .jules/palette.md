## 2026-07-04 - Fix missing ARIA label on nested icon button
**Learning:** When an icon-only interactive element (such as a `<Button>`) is nested inside a wrapper link (`<a>` tag) that provides a `title` or `aria-label`, the innermost interactive element must still have its own `aria-label` attribute to be correctly identified by screen readers.
**Action:** Always ensure nested icon-only interactive elements possess an `aria-label` even if their parent wrapper provides one.
