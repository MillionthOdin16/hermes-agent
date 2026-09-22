## 2026-06-27 - Screen Reader Visibility for Nested Interactive Elements
**Learning:** When an icon-only interactive element (like a `<Button>`) is nested inside a wrapper link (`<a>` tag) that has a `title` or `aria-label`, the innermost interactive element still needs its own `aria-label` attribute. Otherwise, screen readers may focus on the button but fail to announce its purpose.
**Action:** Always verify that the innermost interactive element of any component hierarchy (especially icon-only ones) has an explicit `aria-label`, regardless of the attributes on its parent wrappers.
