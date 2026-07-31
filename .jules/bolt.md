## 2026-07-03 - Memoize RegExp in Inline Markdown Rendering
**Learning:** Inline component render functions like `HighlightedText` are called frequently (especially inside recursive AST evaluations like markdown parsing), which means dynamic variables created inside them (like RegExp built from arrays) are re-evaluated thousands of times unnecessarily, spiking CPU usage.
**Action:** Always wrap dynamically generated regular expressions inside a `useMemo` when they depend on state or props in heavily repeated render components.
