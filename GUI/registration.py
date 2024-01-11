import tkinter as tk

import utils.utils


class RegistrationWindow(tk.Toplevel):
    def __init__(self, login_screen):
        super().__init__()

        self.login_screen = login_screen
        self.title("Register")
        self.geometry(f"{1920}x{1080}")

        # Registration Fields
        self.label_username = tk.Label(self, text="Username:")
        self.label_username.pack(pady=(20, 5))
        self.entry_username = tk.Entry(self)
        self.entry_username.pack(pady=5)

        self.label_password = tk.Label(self, text="Password:")
        self.label_password.pack(pady=(10, 5))
        self.entry_password = tk.Entry(self, show="*")
        self.entry_password.pack(pady=5)

        self.label_email = tk.Label(self, text="Email:")
        self.label_email.pack(pady=(10, 5))
        self.entry_email = tk.Entry(self)
        self.entry_email.pack(pady=5)

        self.label_fullname = tk.Label(self, text="Full Name:")
        self.label_fullname.pack(pady=(10, 5))
        self.entry_fullname = tk.Entry(self)
        self.entry_fullname.pack(pady=5)

        # Register Button
        self.btn_register = tk.Button(self, text="Register", command=lambda: self.registerUser())
        self.btn_register.pack(pady=20)

    def registerUser(self):
        # Get values from the entry fields
        username = self.entry_username.get()
        password = self.entry_password.get()
        email = self.entry_email.get()
        full_name = self.entry_fullname.get()

        utils.utils.register_user(username, password, email, full_name)

        # Close the registration window
        self.destroy()