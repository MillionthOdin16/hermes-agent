## 2026-06-25 - Consolidate multiple re.sub calls
**Learning:** Sequential `re.sub` calls that each remove specific substrings (e.g. `:\d+`, `-v\d+`, `-\d{8}`) can be much slower than a single `re.sub` that uses a combined pattern (e.g. `(?::\d+|-v\d+|-\d{8})+$`). Pre-compiling the regex object also avoids recompilation overhead on hot paths.
**Action:** When finding multiple sequential `re.sub` calls removing string suffix variations, combine them into one compiled regex with an alteration group and a `+$` suffix.
