## 2026-06-12 - Add aria-label to icon-only button inside link
**Learning:** Found an icon-only button nested inside an `<a>` tag that lacked its own `aria-label`. Even if the `<a>` wrapper has a `title`, the innermost interactive element (`<Button>`) must have its own `aria-label` for screen readers to announce it properly.
**Action:** Always add `aria-label` to icon-only interactive elements, especially if they are nested within links or other containers.
