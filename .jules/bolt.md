## 2026-07-17 - Pre-compile Regexes
**Learning:** Python regular expressions used in hot loops or frequently called functions should be pre-compiled at the module or class level to avoid repeated compilation overhead.
**Action:** Lift `re.compile` calls outside of frequently executed functions or loops.
