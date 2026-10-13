## 2026-10-13 - Focus Visible on Language and Theme switchers
**Learning:** Dropdowns like language and theme switchers need clear focus visible rings when navigated via keyboard. Often missed when custom `<Button>` components override default focus styles without providing replacement `focus-visible:` classes, leaving keyboard users with no visual indication of their position.
**Action:** Add `focus-visible:ring-1 focus-visible:ring-current/40` or similar explicit focus rings to standard interactive components that are likely to be accessed via tab navigation.
