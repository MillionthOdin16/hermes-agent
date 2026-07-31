## 2026-07-13 - [Fix ARIA attributes on buttons in a tag]
**Learning:** Found a nested `<Button>` component inside an `<a>` tag (`web/src/components/OAuthProvidersCard.tsx`) which is an icon-only button without its own `aria-label` attribute and receives keyboard focus unnecessarily due to the parent link structure.
**Action:** Always provide explicit `aria-label` and `tabIndex={-1}` for an interactive element (like an icon-only Button) even when wrapped by a focusable tag with accessibility features.
