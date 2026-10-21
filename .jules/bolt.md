## 2026-10-21 - [Optimize CJK char detection in FTS index]
**Learning:** Using `any(...)` with a generator expression inside high-frequency character parsing functions (like `_is_cjk` for FTS search indexing) introduces significant generator overhead and acts as a hidden performance bottleneck.
**Action:** Replace `any()` + generator patterns with standard explicit `for` loops in critical inner loops (e.g., checking CJK character ranges) to remove the overhead and achieve ~4x speedups.
