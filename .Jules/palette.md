## 2026-06-28 - Missing ARIA label on icon-only button inside anchor wrapper
**Learning:** When an icon-only interactive element (like a Button) is nested inside a wrapper link (like an <a> tag) that has a 'title' or 'aria-label', screen readers might still fail to read the inner button properly if it is independently focusable or interactable. It needs its own aria-label.
**Action:** Always ensure that innermost interactive elements (like icon-only Buttons) have their own explicit aria-label attribute, even if their parent wrapper provides context.
