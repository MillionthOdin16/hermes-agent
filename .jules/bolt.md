
## 2026-03-01 - Consolidating Regex Passes for Performance
**Learning:** When processing strings against multiple regex patterns, compiling them into a single regex pass with a capture group is significantly more performant than running `.sub()` in a loop. In this codebase's `strip_think_blocks` (a hot path for text sanitization), multiple loop-based replacements were slowing down text processing. Using `re.compile(rf"<({'|'.join(_REASONING_TAG_NAMES)})>.*?</\1>")` effectively combined these without losing functionality.
**Action:** Avoid loops with `.sub()` when matching multiple literal-based patterns; instead combine them into a single regex pass utilizing `|` for options and `\1` backreferences.
