import subprocess
import os

def test():
    cmd = "echo hello"
    r = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd(),
        stdin=subprocess.DEVNULL,
    )
    return r
