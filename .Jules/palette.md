## 2026-06-02 - Added ARIA label to external link button
**Learning:** Found an icon-only button lacking an `aria-label` inside `OAuthProvidersCard.tsx`, making it inaccessible for screen readers. Added `aria-label` directly to the `Button` element rather than relying on the `title` attribute of the wrapping `<a>` tag.
**Action:** Next time, I will ensure that all icon-only buttons have an `aria-label` explicitly defined on the button element itself, even if wrapped by an element with a title attribute, for better accessibility support.
