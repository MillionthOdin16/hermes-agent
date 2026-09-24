## 2026-09-24 - [Generator Expression Memory Allocation Hoisting]
**Learning:** In heavily used code paths and complex generator comprehensions, evaluating operations that generate new allocations on invariant variables (like `prefix.lower()` or `str(exc).lower()`) over and over on each generator cycle can create unnecessary memory thrash and latency.
**Action:** Always hoist loops invariants out of comprehensions and generators (e.g. `prefix_lower = prefix.lower()` instead of `[mid for mid in catalog if mid.lower().startswith(prefix.lower())]`).
