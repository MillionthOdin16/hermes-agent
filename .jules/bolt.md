## 2026-06-11 - Hoisted Regex compilations in gateway/platforms/signal_format.py
**Learning:** The formatting pipeline converted Markdown to Signal styles by compiling the same set of regular expressions (for bold, italics, code blocks, etc.) on every invocation. This caused unnecessary `re.compile` overhead in a frequently called formatting path for all Signal messages.
**Action:** Always pre-compile static `re.Pattern` objects at the module level rather than repeatedly instantiating them inside frequently executed string processing functions to eliminate cold-cache fan-out penalties.
