## 2026-06-15 - [Add missing aria-label to nested interactive elements]
**Learning:** When an icon-only interactive element (like a Button) is nested inside a wrapper link (like an `<a>` tag) that already has a title, the innermost interactive element itself still needs its own `aria-label` attribute to be properly identified by screen readers, avoiding empty button announcements. This is a common pattern for "external link" buttons.
**Action:** Always add `aria-label` to the inner `<Button size="icon">` even if the parent wrapper `<a>` has a `title` or `aria-label`.
