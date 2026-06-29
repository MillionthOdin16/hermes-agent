## 2026-06-06 - [Nested Link Button Accessibility]
**Learning:** When an icon-only interactive element (like a `<Button>`) is nested inside an `<a>` tag with its own `title` or `aria-label`, screen readers may still focus on the inner button independently and announce it as an unlabeled button.
**Action:** The innermost interactive element (the `<Button>`) must explicitly have its own `aria-label` to be correctly identified by screen readers, even if the parent link has a descriptive `title`.
