import tkinter as tk
from tkinter import messagebox


import requests

import utils.utils
from GUI.registration import RegistrationWindow



class LoginScreen(tk.Tk):
    def __init__(self, main_app):
        super().__init__()

        self.title("Login")
        self.geometry(f"{1920}x{1080}")

        # Login Fields
        self.label_username = tk.Label(self, text="Username:")
        self.label_username.pack(pady=(20, 5))
        self.entry_username = tk.Entry(self)
        self.entry_username.pack(pady=5)

        self.label_password = tk.Label(self, text="Password:")
        self.label_password.pack(pady=(10, 5))
        self.entry_password = tk.Entry(self, show="*")
        self.entry_password.pack(pady=5)

        # Login Button
        self.btn_login = tk.Button(self, text="Login", command=self.login)
        self.btn_login.pack(pady=20)

        # Registration Button
        self.btn_register = tk.Button(self, text="Register", command=self.register)
        self.btn_register.pack(pady=10)

        # Reference to the main application
        self.main_app = main_app

    def login(self):
        # Get username and password from the entry fields
        username = self.entry_username.get()
        password = self.entry_password.get()

        try:
            utils.utils.login_user(username, password)

            messagebox.showinfo("Login Successful", "Welcome, " + username + "!")
            self.destroy()  # Close the login screen
            self.main_app.login_successful = True
        except requests.exceptions.HTTPError as err:
            messagebox.showerror("Login Failed", "Invalid username or password")
            self.main_app.login_successful = False

    def register(self):
        registration_window = RegistrationWindow(self)



