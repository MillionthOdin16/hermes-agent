## 2026-07-17 - Fix nested interactive element accessibility in OAuthProvidersCard
**Learning:** When an icon-only button is nested inside a link (`<a>` tag) that provides context via `title` or `aria-label`, the innermost interactive element (`<Button>`) still needs its own `aria-label` to be correctly identified by screen readers. Furthermore, adding `tabIndex={-1}` to the inner element prevents redundant tab stops.
**Action:** Always provide an explicit `aria-label` and `tabIndex={-1}` to inner icon-only interactive elements when nested in wrapper links.
