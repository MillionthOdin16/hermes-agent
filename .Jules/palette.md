## 2026-06-01 - Icon-Only Buttons Within Anchor Tags
**Learning:** Even when wrapped in an anchor tag with a `title` attribute, icon-only `<Button>` components (like those displaying an `<ExternalLink />`) can still be poorly parsed by screen readers or considered inaccessible unless the button element itself has an explicit `aria-label`.
**Action:** Always ensure that icon-only buttons receive an `aria-label` describing their action, regardless of whether their parent container already provides descriptive text.
