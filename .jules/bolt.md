## 2024-05-30 - Regex compilation in functions
**Learning:** Compiling regex patterns inside functions causes unnecessary overhead every time the function is called.
**Action:** Move regex patterns compiled with `re.compile` to the module or class level when they are static.
