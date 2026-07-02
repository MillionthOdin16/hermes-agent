## 2026-07-02 - Nested Interactive Elements Accessibility
**Learning:** When an icon-only interactive element (like a `<Button>`) is nested inside a wrapper link (`<a>`) that provides a `title` or `aria-label`, screen readers may not reliably identify the nested interactive element if it lacks its own `aria-label`.
**Action:** Always ensure that icon-only interactive elements possess an `aria-label`, even if they are nested within elements that provide accessible information.
