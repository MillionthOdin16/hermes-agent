## 2026-05-19 - Compile Static Regexes at Module Level
**Learning:** Compiling static regular expressions (`re.compile`) inside frequently executed functions incurs measurable performance overhead due to recompilation on every invocation, and cache eviction risks if the global cache fills up.
**Action:** Always move static `re.compile()` calls to the module level or class level, instead of defining them locally inside a function.
