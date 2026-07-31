## 2026-06-27 - Hoist inner regex compilation
**Learning:** Inline compiling `re.compile()` inside frequently called methods introduces unnecessary execution overhead.
**Action:** Hoist static regular expressions to the module or class level to prevent them from being re-compiled on every function invocation.
