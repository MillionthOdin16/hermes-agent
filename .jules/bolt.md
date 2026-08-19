## 2026-10-27 - Consolidate Regex Allocations
**Learning:** In highly trafficked areas like `run_agent.py` and `hermes_state.py`, performing multiple `re.sub` calls sequentially causes excessive intermediate string allocations in Python.
**Action:** When a string needs multiple string substitutions where each matched segment is distinct, use a single `re.sub` combined with alternation `|` and a lambda callback to process matches in a single pass over the string.
