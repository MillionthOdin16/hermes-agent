## 2026-06-29 - Regex compilation hoisting in hot path
**Learning:** Found multiple places in `tools/tts_tool.py` where `re.compile` is called inside functions that might be called repeatedly (e.g., `_render_command_tts_template`, `_compose_gemini_tts_prompt`, `_stream_tts_task`). Compiling regular expressions locally inside functions incurs unnecessary execution overhead.
**Action:** Extract the compiled regex patterns to the module level as constants to prevent redundant compilations on every invocation.
