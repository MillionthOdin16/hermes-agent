## 2026-07-25 - [Accessibility Issue Pattern - Missing aria-label on icon-only buttons inside anchor tag]
**Learning:** We need to keep a watch out for any accessibility issues patterns in our components. When icon-only interactive element is nested inside a wrapper link (\`<a>\`) that provides a \`title\` or \`aria-label\`, the innermost interactive element must still have its own \`aria-label\` attribute to be correctly identified by screen readers, and \`tabIndex={-1}\` to prevent double focus.
**Action:** Be sure to fix them when possible.
