## 2026-06-15 - [Add aria-label to nested interactive elements]
**Learning:** When an icon-only interactive element (like `<Button>`) is nested inside a wrapper link (`<a>`) that already provides a `title` or `aria-label`, the innermost interactive element still must have its own `aria-label` to be correctly identified by screen readers.
**Action:** Always ensure innermost interactive elements (like icon-only buttons) receive an `aria-label`, regardless of what their parent wrappers provide.
