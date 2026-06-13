## 2026-06-13 - [OAuthProvidersCard Docs Link Accessibility]
**Learning:** When nesting an icon-only `Button` inside an `a` tag (link) that already provides a `title` or `aria-label`, the innermost interactive element (`Button`) must still have its own `aria-label` attribute to be correctly identified by screen readers. A missing inner `aria-label` causes accessibility issues.
**Action:** Always verify that innermost interactive components like `Button` have explicit `aria-label`s, even if their wrapper elements provide descriptive text.
