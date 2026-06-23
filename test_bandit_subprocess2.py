import subprocess
import os

def run_command(cmd):
    r = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd(),  # nosec B602
        stdin=subprocess.DEVNULL,
    )
    return r
