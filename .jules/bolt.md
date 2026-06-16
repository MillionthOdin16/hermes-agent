## 2026-01-01 - Avoid redundant regex compilation in hot paths
**Learning:** Found an instance in `gateway/run.py` where a global regular expression (`_TOOL_MEDIA_RE`) was being redefined and recompiled inside a `for` loop over `agent_history`. This recompilation inside a loop causes unnecessary CPU overhead per history message, violating the principle of compiling static regexes at the module level.
**Action:** Always define static `re.compile` patterns at the module/class level. Never compile them inside loops or hot path functions.
