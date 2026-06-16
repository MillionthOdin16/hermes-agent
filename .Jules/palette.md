## 2026-05-21 - Added aria-label to external link icon
**Learning:** Icon-only `<Button>` components nested inside an `<a>` tag with a `title` still require their own `aria-label` attribute to be properly identified by screen readers, as the title on the wrapper doesn't automatically translate to the interactive element inside.
**Action:** Always add an `aria-label` directly to icon-only `<Button>` elements, even when they are wrapped in parent links or containers with descriptive titles.
