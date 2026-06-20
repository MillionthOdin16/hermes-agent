## 2026-10-27 - Hoist compiled regular expressions to module level

**Learning:** When compiling a constant list of regular expressions dynamically inside hot paths like message formatting functions, there is unnecessary CPU overhead on every message processed. Recompiling `re.compile` patterns that contain constant static regex rules dynamically across arrays or loops within functions adds latency over time.
**Action:** Always identify and extract constant regular expressions out of repeatedly executed functions and into module-level globals so they are compiled only once during script startup.
