import subprocess
import time

fastapi_process = subprocess.Popen(["uvicorn", "gui:app", "--host", "127.0.0.1", "--port", "8000"])

# Allow some time for the FastAPI server to start before launching the GUI
time.sleep(2)

subprocess.run(["python", "gui.py"])

fastapi_process.terminate()
