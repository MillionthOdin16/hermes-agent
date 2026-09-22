## YYYY-MM-DD - Icon-only buttons inside links missing individual aria-labels
**Learning:** Even when a wrapper link (`<a>` tag) provides a `title` or `aria-label`, icon-only interactive elements (like a `<Button>`) nested inside it must still have their own `aria-label` attribute to be correctly identified by screen readers, and nesting `<button>` inside `<a>` is technically invalid HTML. However, fixing the missing `aria-label` is a minimal, focused change.
**Action:** Add `aria-label` to icon-only components even when they are inside descriptive wrappers.
