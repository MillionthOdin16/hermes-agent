## 2026-03-24 - Hoist re.compile
**Learning:** Compiling regular expressions inside methods causes them to be re-compiled on every invocation, causing unnecessary execution overhead.
**Action:** Extract `re.compile` calls to module or class level when they define static patterns, preventing execution overhead in hot paths.
