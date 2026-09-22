## 2026-06-25 - Accessibility: Icon-only buttons inside wrapper links need their own aria-label
**Learning:** When an icon-only interactive element (like a Button) is nested inside a wrapper link (a tag) that provides a title or aria-label, the button itself must still have its own aria-label attribute to be correctly identified by screen readers.
**Action:** Always ensure innermost interactive elements have their own aria-label, even if parent elements provide context.
