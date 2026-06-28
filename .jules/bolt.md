
## 2026-06-04 - Module-level Regex Compilation
**Learning:** In highly-frequently executed methods (e.g., search parsers, TTS streamers, and memory extractors), compiling regular expressions using `re.compile` internally introduces measurable overhead as it creates a new Pattern object each time or relies heavily on the internal cache if the cache hasn't evicted it.
**Action:** Always move static `re.compile` declarations to the module or class level to avoid recurrent instantiation and to improve performance, especially on critical data paths.
