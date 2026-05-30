## 2026-05-30 - Added ARIA labels to ChatSidebar buttons
**Learning:** Re-emphasized the importance of adding `aria-label` attributes to icon-only buttons or buttons where the textual meaning is primarily visual. Adding an aria-label matching the text content, while sometimes redundant, is completely safe and guarantees proper reading by screen readers.
**Action:** Consistently search for `prefix`, `suffix`, or `icon` only elements inside `Button` components across all `tsx` files, and ensure an explicit `aria-label` exists for accessibility.
