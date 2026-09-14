## 2026-10-24 - Hoist loop-invariant method calls in generator expressions
**Learning:** Calling `.lower()` inside a Python generator expression (e.g., `any(marker in text.lower() for marker in list)`) re-evaluates and allocates a new string object on *every single iteration* of the generator.
**Action:** Always extract loop-invariant operations like string conversions outside of comprehensions and generator expressions, especially in error-handling hot paths where the same variable is repeatedly checked against multiple markers.
