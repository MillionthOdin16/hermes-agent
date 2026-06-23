import subprocess
import os

def run_command(cmd):
    r = subprocess.run(  # nosec B602
        cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd(),
        stdin=subprocess.DEVNULL,
    )
    return r
