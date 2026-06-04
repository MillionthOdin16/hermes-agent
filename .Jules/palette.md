## 2026-02-13 - Nested Interactive Elements Accessibility
**Learning:** When an icon-only interactive element (such as a `<Button>`) is nested inside a wrapper link (`<a>` tag) that provides a `title` or `aria-label`, screen readers still require the innermost interactive element to have its own `aria-label` attribute to be correctly identified.
**Action:** Always add an `aria-label` directly to the `<Button>` or inner interactive element itself, even if a parent element already has a `title` or `aria-label`.
