## 2026-07-06 - Nested Interactive Elements inside Wrapper Links
**Learning:** When an icon-only interactive element (like a `<Button>`) is nested inside an `<a>` tag that provides a `title` or `aria-label`, the innermost interactive element still needs its own `aria-label` attribute to be correctly identified by screen readers, and should be explicitly removed from the tab sequence with `tabIndex={-1}` to prevent redundant tab stops.
**Action:** Always add an `aria-label` and `tabIndex={-1}` to buttons nested within wrapper links.
