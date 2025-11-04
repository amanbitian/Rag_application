import os
import subprocess
from pathlib import Path

# Ensure working directory is project root
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# Command to run Streamlit app
cmd = ["streamlit", "run", "ui/app.py"]

print("🚀 Launching RAG Corp App...\n")
subprocess.run(cmd, check=True)


### Working model for test pulled from dev on 4th nov