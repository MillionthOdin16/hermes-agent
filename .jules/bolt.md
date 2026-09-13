## 2026-03-24 - Hoist string lowering out of generators to prevent repeated allocations
**Learning:** Python generators evaluating in loops (like `any(...)`) re-evaluate method calls on the enclosing scope (like `cause.lower()` or `str(exc).lower()`) on every single iteration. This repeatedly creates new string allocations, causing performance regressions in tight loops or large error bodies (like in `hermes_state.py`).
**Action:** Always hoist `.lower()` calls and string conversions outside of generator expressions and comprehensions to execute exactly once.
