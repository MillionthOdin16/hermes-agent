## 2026-06-17 - Missing ARIA label on icon-only button within a link
**Learning:** Even when a wrapper `<a>` tag has a `title` or `aria-label`, screen readers often announce the innermost interactive element (like a `<button>`). If that element only contains an icon and lacks its own `aria-label`, the screen reader may fail to convey its purpose.
**Action:** Always add an `aria-label` to icon-only `<Button>` or `<button>` elements, regardless of whether they are wrapped in an anchor tag with descriptive attributes.
