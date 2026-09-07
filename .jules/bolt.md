## 2026-06-15 - Consolidating regex tag parsing
**Learning:** Compiling and looping through multiple separate regexes to strip XML-like tags (e.g. `<think>`, `<tool_call>`) adds significant string processing overhead on large LLM outputs.
**Action:** Consolidate sequential `.sub()` passes into a single pre-compiled regex utilizing capturing groups and backreferences (e.g. `r'<(tag1|tag2)>.*?</\1>'`) to improve parsing performance.
