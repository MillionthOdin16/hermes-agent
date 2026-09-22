## 2026-06-06 - [Aria label on icon-only buttons in links]
**Learning:** [When an icon-only interactive element (like `<Button>`) is nested inside a wrapper link (`<a>` tag) that provides a `title` or `aria-label`, the innermost interactive element must still have its own `aria-label` attribute to be correctly identified by screen readers.]
**Action:** [Always ensure that all interactive elements, like buttons, have an `aria-label` when they only contain an icon, even if they are wrapped by another element with an accessible name.]
