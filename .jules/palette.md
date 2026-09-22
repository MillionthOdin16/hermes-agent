## 2026-07-09 - Accessibility of buttons inside wrapper links
**Learning:** When an icon-only interactive element (like `<Button>`) is nested inside a wrapper link (`<a>` tag) that provides a `title` or `aria-label`, the innermost element must still have its own `aria-label` attribute to be correctly identified by screen readers. Additionally, it should be explicitly removed from the tab sequence with `tabIndex={-1}` to prevent redundant tab stops.
**Action:** Always add an explicit `aria-label` and `tabIndex={-1}` to inner interactive elements when they are nested inside wrapper links.
