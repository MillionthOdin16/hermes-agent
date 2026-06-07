## 2024-05-30 - Added ARIA label to OAuthProviderCard Docs Button
**Learning:** Icon-only buttons used for external links using the lucide-react components (e.g. `<Button><ExternalLink /></Button>`) inside this app often lacked `aria-label` because the wrapper `<a title="...">` was relied on, but a native `aria-label` is better.
**Action:** Consistently apply `aria-label` on `<Button size="icon">` elements across components even if their wrappers provide some hover context, to improve screen reader accessibility.
