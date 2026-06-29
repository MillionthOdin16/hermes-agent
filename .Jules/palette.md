
## 2026-06-08 - Added ARIA Label to OAuth Docs Button
**Learning:** Screen readers might skip the `title` attribute of an `<a>` tag or associate it differently. When nesting an icon-only `<Button>` within a link, the button element itself should carry an `aria-label` to ensure unambiguous accessibility, especially for components passing through multiple DOM layers.
**Action:** Always provide explicit `aria-label`s directly on `<Button>` components when they only render icons, even if their wrapping `<a>` tags contain `title`s or other descriptive text.
