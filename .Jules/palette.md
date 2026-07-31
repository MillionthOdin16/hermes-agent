## 2026-05-15 - Missing ARIA labels on nested interactive elements
**Learning:** Icon-only interactive elements (`<Button>`) nested inside wrapper links (`<a>`) rely on the wrapper’s `title` attribute, but screen readers may fail to announce the button correctly if the innermost interactive element lacks its own `aria-label`. This pattern exists in `OAuthProvidersCard.tsx`.
**Action:** Always provide an explicit `aria-label` on the innermost interactive element (like `<Button>`), even if its parent wrapper (`<a>`) already provides a `title` or `aria-label`.
