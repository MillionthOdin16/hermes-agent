## 2026-06-29 - Missing ARIA Labels on Icon Buttons within Links
**Learning:** When icon-only interactive elements (like a `<Button ghost size="icon">` using an `<ExternalLink />` icon) are nested inside wrapper links (`<a>` tags) that have a `title` attribute, the innermost `<Button>` may still lack its own `aria-label`. Screen readers often depend on the innermost interactive element to announce its purpose.
**Action:** Always ensure that icon-only `<Button>` elements receive an `aria-label`, even if their parent wrapper has a `title` or `aria-label`.
