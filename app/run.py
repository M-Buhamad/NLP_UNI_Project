#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT= Path(__file__).resolve().parent
APP_PATH= PROJECT_ROOT / "app.py"


def main():
    if not APP_PATH.is_file():
        raise FileNotFoundError(f"Streamlit app not found: {APP_PATH}")

    command= [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
    ]

    subprocess.run(command, cwd= PROJECT_ROOT, check= True)


if __name__ == "__main__":
    main()