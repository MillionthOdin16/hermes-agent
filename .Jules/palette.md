## 2026-06-25 - Missing ARIA labels on icon-only buttons
**Learning:** Icon-only buttons (using `size="icon"`) often lack `aria-label` attributes, making them inaccessible to screen readers. In `web/src/components/OAuthProvidersCard.tsx`, an external link icon to documentation was completely silent to screen readers.
**Action:** When adding `size="icon"` buttons, always test accessibility. For external links, `aria-label={"Open " + name + " docs"}` ensures context is preserved.
