## 2026-05-24 - [Avoid re.compile in Hot Loops]
**Learning:** Re-compiling regular expressions inside tight loops (like iterating over message history) introduces unnecessary overhead and scales poorly with the number of messages. Python's re.compile is intended to be initialized once, ideally at module level.
**Action:** Always declare `re.compile` at the module or class level to avoid redundant instantiation in frequently executed code paths.
