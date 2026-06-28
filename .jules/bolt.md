## 2026-06-03 - Compile regex at module level
**Learning:** Using `re.compile()` inside loops or functions adds significant overhead (approx 50% slower per the benchmark).
**Action:** Always compile static regular expressions at the module or class level to prevent unnecessary function call execution overhead.
