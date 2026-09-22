## 2026-09-16 - Add ARIA label to Model Picker search input
**Learning:** The Model Picker dialog's search input lacked an `aria-label`, making it less accessible for screen reader users since it doesn't have an explicit `<label>`.
**Action:** Always add an `aria-label` to standalone search inputs or use an explicitly associated `<label>` element.
