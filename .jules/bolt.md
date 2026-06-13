## 2026-06-13 - [Extract inner functions to module level in Python]
**Learning:** Python re-evaluates inner functions every time the outer function is called. When called in a tight loop, this overhead becomes significant. `agent/redact.py` called several such inner functions inside `redact_sensitive_text`, taking ~2.6s for 100k invocations.
**Action:** Extract nested `def`s passed as `re.sub` callbacks into module-level functions where possible. This improves speed (down to ~1.9s for 100k calls).
