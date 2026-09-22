## 2026-07-15 - Nested Interactive Element Accessibility
**Learning:** When using custom components like `<Button>` inside wrapper links (`<a>`) that handle navigation, the nested button can create redundant tab stops or missing context for screen readers if it lacks its own ARIA label and `tabIndex={-1}`.
**Action:** Always apply `aria-label` and `tabIndex={-1}` to icon-only buttons when nested inside link tags that already provide the title/label, preventing double-focus issues while maintaining accessibility.
