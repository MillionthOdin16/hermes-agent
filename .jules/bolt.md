## 2026-03-06 - Consolidate multiple re.sub passes
**Learning:** Performing multiple sequential `re.sub` passes to find or replace similar string patterns (e.g., stripping various reasoning tags) adds parsing overhead. Consolidating them into a single pass using capturing groups and backreferences significantly improves string processing performance.
**Action:** When applying multiple regex replacements for similar tag structures, consolidate them into a single regex with backreferences.
