## 2026-05-18 - Nested Button Accessibility
**Learning:** When an icon-only interactive element (like `<Button>`) is nested inside a wrapper link (`<a>` tag) that provides a `title` or `aria-label`, the innermost interactive element must still have its own `aria-label` attribute to be correctly identified by screen readers.
**Action:** Always verify that nested interactive elements have their own `aria-label` attributes, even if the parent container provides context.
