## 2026-10-27 - re.compile overhead
**Learning:** Found several places where `re.compile` is called inside loops or functions instead of module level, causing overhead.
**Action:** Move `re.compile` calls to module-level constants to avoid recompilation overhead on each invocation.
