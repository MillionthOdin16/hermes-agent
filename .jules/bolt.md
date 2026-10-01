## 2026-10-01 - Optimizing Context Compressor
**Learning:** Found an expensive generator inside a dictionary comprehension in `agent/context_compressor.py`. The `skill.lower()` string method was being called inside `any(skill.lower() in text for text in tail_user_texts)` for each message in the tail, evaluating repeatedly inside the loop.
**Action:** Always hoist string conversions (like `.lower()`) and other loop-invariant evaluations outside of generator expressions and loops to avoid redundant, expensive allocations and speed up execution, particularly for large iterative structures.
