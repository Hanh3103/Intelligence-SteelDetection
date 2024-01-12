import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

import requests

import utils.utils
from GUI.registration import RegistrationWindow



class LoginScreen(ctk.CTk):
    def __init__(self, main_app):
        super().__init__()

        self.title("Login")
        self.geometry(f"{400}x{400}")
        #Caption
        label = ctk.CTkLabel(self, text="Anmeldung Stahlklassifizierung")
        label.grid(row=0, column=0)

        # Login Fields

        self.entry_username = ctk.CTkEntry(self, placeholder_text="Username")
        self.entry_username.grid(row=1, column=0, pady=5)
        self.entry_password = ctk.CTkEntry(self, show="*", placeholder_text="Password")
        self.entry_password.grid(row=2, column=0, pady=5)


        # Login Button
        self.btn_login = ctk.CTkButton(self, text="Login", command=self.login)
        self.btn_login.grid(row=3, column=0, pady=5)
        self.bind('<Return>', lambda event: self.login())


        # Registration Button
        self.btn_register = ctk.CTkButton(self, text="Register", command=self.register)
        self.btn_register.grid(row=4, column=0, pady=5)

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



