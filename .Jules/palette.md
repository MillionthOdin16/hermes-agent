## 2026-03-01 - Missing ARIA Labels on Icon-only Buttons
**Learning:** Found multiple instances where `<Button ghost size="icon">` was used without an accompanying `aria-label`, specifically for external links (e.g., `ExternalLink` icon). This is a recurring pattern when nesting icons inside buttons.
**Action:** When wrapping an icon in a `Button` with `size="icon"`, always explicitly add an `aria-label` to the `Button` component, even if the surrounding anchor tag has a `title`.
