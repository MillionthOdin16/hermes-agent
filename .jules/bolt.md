## 2026-07-09 - Optimize reasoning tag stripping overhead

**Learning:** String processing functions like `_strip_reasoning_tags` in the frontend formatting pipeline are called repeatedly on each message rendering step. Using inline `re.sub()` inside loops dynamically re-compiles complex regular expressions. Implementing a fast-path early return (e.g. checking if `<` is in the string) and pre-compiling the regexes at the module level provided a >10x speedup (0.46s to 0.03s in tests).
**Action:** Always pre-compile regular expressions at the module scope if they are evaluated within hot paths or loops, and implement fast-path early exits for string manipulation when the target character/substring isn't even present.
