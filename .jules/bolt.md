## 2026-05-19 - Hoist regular expressions to avoid recompilation
**Learning:** The Python Code Convention (Performance) advises pre-compiling static regular expressions (`re.compile`) at the module or class level rather than inside frequently called functions (like formatting adapters) to prevent redundant compilation overhead.
**Action:** Always extract `re.compile` patterns to the module level.
