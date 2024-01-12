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
        self.geometry(f"{400}x{350}")
        self.center_window()

        #Caption
        label = ctk.CTkLabel(self, text="Anmeldung Stahlklassifizierung", font=("Helvetica", 16, "bold"))
        label.grid(row=0, column=0, sticky="ew", pady=40, padx=80)

        # Entry widgets with placeholders
        self.entry_username = ctk.CTkEntry(self, placeholder_text="Username")
        self.entry_username.grid(row=1, column=0, pady=5, padx=100, sticky="ew")
        self.entry_password = ctk.CTkEntry(self, show="*", placeholder_text="Password")
        self.entry_password.grid(row=2, column=0, pady=5, padx=100, sticky="ew")

        # Login Button
        self.btn_login = ctk.CTkButton(self, text="Login", command=self.login)
        self.btn_login.grid(row=3, column=0, pady=5, padx=100, sticky="ew")
        self.bind('<Return>', lambda event: self.login())

        # Registration Button
        self.btn_register = ctk.CTkButton(self, text="Register", command=self.register)
        self.btn_register.grid(row=4, column=0, pady=1, padx=100, sticky="ew")

        # Center the widgets vertically and horizontally
        #self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

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

    def center_window(self):
        # Get the screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate the window position
        x = (screen_width - self.winfo_reqwidth()) // 2
        y = (screen_height - self.winfo_reqheight()) // 2

        # Set the window position
        self.geometry("+{}+{}".format(x, y))



