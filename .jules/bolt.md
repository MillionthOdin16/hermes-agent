## 2026-05-15 - Compile Regex at Class Level
**Learning:** Compiling regular expressions repeatedly inside frequently called functions, like `_auto_extract_facts` in `plugins/memory/holographic/__init__.py`, incurs unnecessary overhead, especially as memory interactions occur often.
**Action:** Always extract `re.compile` calls out of functions or methods and store them as class-level or module-level constants to avoid redundant compilation.
