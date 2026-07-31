## 2026-07-08 - Icon-only buttons within wrapper links require specific ARIA properties
**Learning:** Screen readers may fail to announce icon-only buttons properly if they are nested inside wrapper links (`<a>` tags) without their own `aria-label`, even if the wrapper has a `title` or `aria-label`. Furthermore, this pattern creates redundant tab stops.
**Action:** Ensure innermost interactive elements inside wrapper links have an explicit `aria-label` and `tabIndex={-1}` to prevent redundant tab stops and ensure correct screen reader behavior.
