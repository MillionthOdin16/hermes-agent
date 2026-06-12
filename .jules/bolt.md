## 2026-06-12 - Move regex compilation to class/module scope
**Learning:** Compiling regular expressions using `re.compile()` inside loops or functions causes overhead because it recompiles the regex on every invocation, hurting performance, especially in hot paths.
**Action:** Extract compiled regular expressions (`re.compile(...)`) to module-level variables or class-level constants to ensure they are compiled exactly once at load time, which matches optimal codebase-specific performance patterns.
