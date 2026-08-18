## 2026-08-18 - Consolidate Regex Passes
**Learning:** Sequential matching and substitution of related tags using distinct regexes introduces unnecessary loop and object instantiation overhead, causing significant slow-downs when performing dense-char counting.
**Action:** Always combine matching related target tokens into a single unified `re.compile()` using the regex `|` operator for a single regex evaluation pass.
