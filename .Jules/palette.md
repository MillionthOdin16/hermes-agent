## 2026-06-10 - Inner Icon Button ARIA Labels
**Learning:** When an icon-only `<Button>` component is wrapped in an `<a>` link tag that already has a `title` or `aria-label`, screen readers still need the innermost interactive element (the Button itself) to have its own `aria-label`. Otherwise, they might read the button as an unlabeled button element inside the link.
**Action:** Always ensure nested interactive elements have explicit labels, even when their parent wrapper provides context.
