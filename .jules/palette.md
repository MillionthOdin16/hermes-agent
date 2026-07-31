## 2026-07-17 - Add ARIA label and tabIndex to nested icon-only button
**Learning:** When an icon-only `<Button>` is nested inside an `<a>` wrapper link, the inner button lacks an accessible name for screen readers, and causes redundant tab stops.
**Action:** Add `aria-label` to the inner `<Button>` to match the wrapper's `title`, and set `tabIndex={-1}` to prevent redundant tab stops.
