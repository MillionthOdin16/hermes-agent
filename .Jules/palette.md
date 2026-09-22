## 2026-05-31 - Missing Aria Labels in Dashboard
**Learning:** Found an icon-only button without an `aria-label` attribute in `web/src/components/OAuthProvidersCard.tsx`. Although other icon buttons in the repository generally have aria-labels (like in `ConfigPage.tsx`), there are cases where they are missed, specifically in `ExternalLink` usages.
**Action:** Always check `Button size="icon"` components to ensure they have descriptive `aria-label`s, especially when they act as generic external links.
