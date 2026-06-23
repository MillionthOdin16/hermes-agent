## 2026-06-23 - Add ARIA label to nested icon buttons
**Learning:** In the `OAuthProvidersCard`, icon-only `<Button>` elements nested inside wrapper links (`<a>` tags) with `title` attributes were missing their own `aria-label`. Screen readers need the innermost interactive element to have an accessible name.
**Action:** When nesting interactive components inside wrapper links, always verify that the innermost component has an explicit `aria-label` even if the wrapper provides context via `title` or `aria-label`.
