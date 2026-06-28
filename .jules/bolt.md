## 2026-05-18 - [Extract Repeated Regexes]
**Learning:** Found repetitive `re.compile` calls inside tight loops and frequently called methods like `_search_with_rg` and `_search_with_grep` in `tools/file_operations.py`. This wastes CPU cycles on regex compilation on every search call. Moving it to the module level avoids unnecessary compilation overhead.
**Action:** Extract repeated local regex compilations to the module level.
