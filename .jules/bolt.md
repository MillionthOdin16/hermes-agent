## 2026-03-05 - Module-level regex compilation for performance
**Learning:** Static regular expressions (e.g. `_think_block_re = re.compile(...)`) placed inside functions are re-compiled on every function invocation, leading to unnecessary execution overhead in high-frequency paths (like text streaming or file line parsing).
**Action:** Always compile static regex patterns at the module or class level to leverage Python's single-pass compilation, and reference them globally within methods.
