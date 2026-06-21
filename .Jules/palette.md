## 2026-06-21 - Icon-only buttons nested in wrapper links
**Learning:** Screen readers may fail to announce the action of an icon-only interactive element (like `<Button>`) if it lacks its own `aria-label`, even when nested inside a wrapper link (`<a>`) that has a `title` or label.
**Action:** Always verify that innermost interactive UI elements, especially icon-only buttons within external links, possess an explicit `aria-label` attribute to ensure proper accessibility identification independently of their parent context.
