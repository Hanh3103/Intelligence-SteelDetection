import threading
import uvicorn

from fastAPI import fast_app
from gui import App


def run_fastapi_server():
    uvicorn.run(fast_app, host="127.0.0.1", port=8000, log_level="info")


def run_tkinter_app():
    app = App()
    app.mainloop()


def main():
    # Start FastAPI server in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi_server)
    fastapi_thread.start()

    #run_tkinter_app()

if __name__ == "__main__":
    main()
