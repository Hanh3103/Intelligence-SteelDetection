# Import necessary modules
import threading
from fastapi import FastAPI
import uvicorn

# Import components from custom modules
from GUI.gui import App
from GUI.login import LoginScreen
from endpoints import endpoints_unsupervised, endpoints_supervised, endpoints_database, endpoints_register_token

# Create a FastAPI instance
fast_app = FastAPI()

# Include routers from different modules
fast_app.include_router(endpoints_unsupervised.router)
fast_app.include_router(endpoints_supervised.router)
fast_app.include_router(endpoints_database.router)
fast_app.include_router(endpoints_register_token.router)


# Function to run the FastAPI server using uvicorn
def run_fastapi_server(app):
    uvicorn.run(fast_app, host="127.0.0.1", port=8000, log_level="info")


# Function to run the Tkinter GUI
def run_tkinter_app(app):
    app.main_app = App()
    app.main_app.mainloop()


# Custom class extending App with login capabilities
class AppWithLogin(App):
    def __init__(self):
        self.login_successful = False
        self.login_screen = None

    def wait_for_login(self):
        # Create and display the login screen
        self.login_screen = LoginScreen(self)
        self.login_screen.mainloop()


# Main function
def main():
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

    # Ensure the FastAPI thread has finished before exiting
    fastapi_thread.join()


# Entry point of the program
if __name__ == "__main__":
    main()
