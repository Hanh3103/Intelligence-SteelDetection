# Import necessary modules
import threading  # Threading module for running FastAPI server in a separate thread
from fastapi import FastAPI  # FastAPI framework for building APIs
import uvicorn  # Uvicorn for running FastAPI server

# Import components from custom modules
from GUI.gui import App  # Custom GUI application
from GUI.login import LoginScreen  # Login screen component
from endpoints import endpoints_unsupervised, endpoints_supervised, endpoints_database, endpoints_register_token
from fastapi import HTTPException, status

# Create a FastAPI instance
fast_app = FastAPI()

# Include routers from different modules
fast_app.include_router(endpoints_unsupervised.router)
fast_app.include_router(endpoints_supervised.router)
fast_app.include_router(endpoints_database.router)
fast_app.include_router(endpoints_register_token.router)


# Function to run the FastAPI server using uvicorn
def run_fastapi_server(app):
    """
    Function to run the FastAPI server using uvicorn.

    Args:
        app: FastAPI instance.

    Returns:
        None
    """
    try:
        uvicorn.run(fast_app, host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error running FastAPI server: {str(e)}")


# Function to run the Tkinter GUI
def run_tkinter_app(app):
    """
    Function to run the Tkinter GUI.

    Args:
        app: Tkinter application instance.

    Returns:
        None
    """
    app.main_app = App()
    app.main_app.mainloop()


# Custom class extending App with login capabilities
class AppWithLogin(App):
    def __init__(self):
        """
        Constructor for AppWithLogin class.

        Initializes login_successful and login_screen attributes.

        Args:
            None

        Returns:
            None
        """
        self.login_successful = False
        self.login_screen = None

    def wait_for_login(self):
        """
        Method to wait for user login.

        Creates and displays the login screen.

        Args:
            None

        Returns:
            None
        """
        self.login_screen = LoginScreen(self)
        self.login_screen.mainloop()


# Main function
def main():
    """
    Main function to run the program.

    Creates the main application instance with login capabilities.
    Starts FastAPI server in a separate thread.
    Waits for login.
    Runs the Tkinter app if the login is successful.

    Args:
        None

    Returns:
        None
    """
    try:
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
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


# Entry point of the program
if __name__ == "__main__":
    main()
