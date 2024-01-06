import threading
import uvicorn

from api.fastAPI import fast_app
from GUI.gui import App


def run_fastapi_server():
    # Function to run the FastAPI server using uvicorn
    uvicorn.run(fast_app, host="127.0.0.1", port=8000, log_level="info")


def run_tkinter_app():
    # Function to run the Tkinter gui
    app = App()
    app.mainloop()


def main():
    # Main function
    # Start FastAPI server in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi_server)
    fastapi_thread.start()

    # Run the Tkinter app
    run_tkinter_app()


if __name__ == "__main__":
    # Entry point of the program
    main()
