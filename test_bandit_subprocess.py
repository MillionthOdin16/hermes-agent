import subprocess
import os

def run_command(cmd):
    # nosec B602
    r = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd(),
        stdin=subprocess.DEVNULL,
    )
    return r
