## 2026-08-04 - Add ARIA Roles to Dropdown Menus
**Learning:** In the `web` workspace UI, custom dropdown menus created with `div` containers and inner `button` elements (like `UseAsMenu` in `ModelsPage.tsx`) require explicit ARIA roles (`role="menu"` on the container and `role="menuitem"` on the children) to ensure screen readers correctly interpret the interaction pattern.
**Action:** Always verify that hand-rolled dropdown implementations include proper `role="menu"` and `role="menuitem"` attributes, along with `aria-haspopup` and `aria-expanded` on the trigger buttons.
