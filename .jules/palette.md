## 2026-10-27 - [Add aria-label and tabIndex={-1} for button inside a tag]
**Learning:** In React, placing a `<button>` inside an `<a>` tag with a `title` or `aria-label` requires the inner button to have its own `aria-label` and `tabIndex={-1}` for correct accessibility and tab-flow.
**Action:** When an interactive icon button element (like `<Button>`) is nested inside an `<a>` tag with accessibility information, apply `aria-label` and `tabIndex={-1}` to the inner button to avoid redundant tab stops and ensure proper screen reader support.
