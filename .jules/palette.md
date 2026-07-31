## 2026-07-09 - Accessible nested icon buttons
**Learning:** When an icon-only interactive element like `<Button>` is nested inside an `<a>` tag that provides a `title` or `aria-label`, the innermost interactive element needs its own `aria-label` for screen readers and `tabIndex={-1}` to prevent redundant tab stops.
**Action:** Always provide explicit `aria-label` and `tabIndex={-1}` to child `<Button>` components wrapped by a link tag.
