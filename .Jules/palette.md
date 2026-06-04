## 2026-03-02 - Add aria-label to external link icon buttons
**Learning:** Icon-only interactive elements (like a button) that are nested inside a wrapper link (`<a>` tag) with `title` or `aria-label` still require an explicit `aria-label` attribute on the innermost interactive element itself to ensure correct identification by screen readers.
**Action:** Always verify that innermost icon-only buttons include `aria-label` attributes even if their parent wrappers have `title` or `aria-label`.
