import threading

import uvicorn

from api.fastAPI import fast_app
from gui.gui import App
from gui.login import LoginScreen


def run_fastapi_server(app):
    # Function to run the FastAPI server using uvicorn
    uvicorn.run(fast_app, host="127.0.0.1", port=8000, log_level="info")


def run_tkinter_app(app):
    # Function to run the Tkinter gui
    app.main_app = App()
    app.main_app.mainloop()


class AppWithLogin(App):
    def __init__(self):
        self.login_successful = False
        self.login_screen = None

    def wait_for_login(self):
        # Create and display the login screen
        self.login_screen = LoginScreen(self)
        self.login_screen.mainloop()


def main():
    # Main function
    # Create the main application instance with login capabilities
    app = AppWithLogin()

    # Start FastAPI server in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi_server, args=(app,))
    fastapi_thread.start()

    # Wait for login
    app.wait_for_login()

    # Run the Tkinter app if the login is successful
    if app.login_successful:
        run_tkinter_app(app)
    else:
        print("Login failed. Exiting...")


if __name__ == "__main__":
    # Entry point of the program
    main()
