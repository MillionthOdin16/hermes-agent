## 2026-07-28 - Add aria-label to icon-only buttons nested in wrapper elements
**Learning:** When an icon-only button is nested inside an `<a>` tag that provides a title, the innermost interactive `<Button>` element must still have an `aria-label` attribute to ensure screen readers correctly identify it, and redundant tab stops are removed using `tabIndex={-1}`.
**Action:** Add `aria-label` and `tabIndex={-1}` to icon-only buttons wrapped in `<a>` tags.
