## 2026-07-07 - Add icon and handle collapsed state for LanguageSwitcher
**Learning:** UI components in collapsible areas (like sidebars) need to support both expanded (text+icon) and collapsed (icon only) states to avoid text truncation and layout issues when the container shrinks.
**Action:** When adding components to a responsive sidebar, verify they accept a `collapsed` prop and gracefully switch to an icon-only representation.
