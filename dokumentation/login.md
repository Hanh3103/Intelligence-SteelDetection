# LoginScreen Documentation

The `LoginScreen` class, derived from `customtkinter.CTk`, provides a user interface for the login and registration functionality within an application.

## Features

- **User Login**: Allows users to log into the application by entering their username and password.
- **User Registration**: Offers a registration form for new users to sign up by providing their username, password, email, and full name.
- **Flexible UI**: Switches between login and registration modes based on user interaction, enhancing user experience.

## Initialization

- `main_app`: Reference to the main application class to access and modify the application state based on the login status.

## Layout

- The UI consists of input fields for username, password, email, and full name, along with corresponding buttons for actions like login, registration, and switching between modes.
- Utilizes `customtkinter` for a modern, themed appearance.

## Key Methods

- `registration_mode()`: Adjusts the UI to display registration fields and buttons.
- `login_mode()`: Resets the UI to the default login view.
- `login()`: Validates user credentials. On success, it closes the login window and updates the main application's login state. On failure, displays an error message.
- `register()`: Captures user input from the registration fields to create a new account. Displays a success message upon successful registration and returns to the login view.
- `center_window()`: Centers the login window on the screen.
- `on_close()`: Custom protocol for the window close event, ensuring a graceful shutdown of the application and related processes.

## Future Work

- **Enhanced Validation**: Implement client-side validation for input fields to provide immediate feedback on user input errors before submission.
- **Password Recovery**: Add a password recovery feature to assist users who forget their passwords.
- **Two-Factor Authentication**: Implement two-factor authentication for additional security during the login process.
