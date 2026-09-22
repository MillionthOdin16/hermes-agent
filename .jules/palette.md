## 2026-08-21 - LanguageSwitcher Sidebar Consistency
**Learning:** The LanguageSwitcher component wasn't adapting its styling/sizing when collapsed like the ThemeSwitcher, leading to an inconsistent sidebar UI pattern.
**Action:** Always check both expanded and collapsed responsive states for sidebar toggle elements, ensuring they share the same 'icon' sizing fallback.
