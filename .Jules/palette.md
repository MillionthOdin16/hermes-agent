## 2024-05-15 - Missing ARIA label on ExternalLink Button
**Learning:** Found an icon-only button (ExternalLink) inside an anchor tag (`<a>`) in `web/src/components/OAuthProvidersCard.tsx` lacking an `aria-label`. Even when nested inside a link with a `title`, the nested interactive element (button) must have its own `aria-label` attribute to be accessible to screen readers, per the accessibility conventions for icon-only buttons.
**Action:** Add `aria-label` to all nested icon-only buttons, even when the parent element provides some accessible name or title.
