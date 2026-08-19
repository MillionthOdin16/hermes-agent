import os

with open("tests/run_agent/test_run_agent.py", "r", encoding="utf-8") as f:
    content = f.read()

print("Number of regex matches found in test_run_agent.py: ", content.count("re.sub"))
