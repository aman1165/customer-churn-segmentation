import subprocess
import time
import webbrowser


subprocess.Popen(
    [
        "uvicorn",
        "app.main:app",
        "--reload",
    ]
)

time.sleep(3)

webbrowser.open("http://127.0.0.1:8000")
