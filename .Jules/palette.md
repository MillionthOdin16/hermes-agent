## 2026-06-23 - Nested interactive elements require direct ARIA labels
**Learning:** Even if a wrapper element like an `<a>` tag has a `title` attribute, any nested interactive element like a `<Button>` component still needs its own `aria-label` to be reliably announced by screen readers.
**Action:** Always verify that innermost icon-only interactive components have their own `aria-label` props, regardless of parent attributes.
