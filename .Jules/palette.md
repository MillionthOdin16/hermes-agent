## 2026-06-02 - Added ARIA label to docs link button in OAuth providers card
**Learning:** When wrapping an icon-only interactive component (like `<Button>`) inside an `<a>` tag that provides a `title`, the inner component still lacks an accessible name for screen readers, leading to confusing or unlabelled interactive elements.
**Action:** Always ensure the innermost interactive element (e.g. `<Button>`) has its own `aria-label`, even if an outer wrapper link has a `title` or `aria-label`.
