## 2026-06-07 - Add aria-label to innermost interactive elements
**Learning:** Even if an outer wrapper element like an `<a>` tag has an accessibility attribute such as `title` or `aria-label`, the innermost interactive element (such as an icon-only `<Button>`) must still possess its own `aria-label`. Without it, screen readers may misidentify or incorrectly announce the interactive part of the component.
**Action:** Always ensure that icon-only interactive components (`Button`, `IconButton`, etc.) have an `aria-label` attribute, regardless of the attributes on their parent wrappers.
