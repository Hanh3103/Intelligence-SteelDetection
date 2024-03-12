from tkinter import messagebox
import customtkinter as ctk

from GUI.gui import stop_fastapi_server
from api_functions.login_register import login_user, register_user


class LoginScreen(ctk.CTk):
    def __init__(self, main_app):
        super().__init__()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.title("Login")
        self.geometry(f"{400}x{350}")
        self.center_window()
        self.grid_columnconfigure(0, weight=1)

        # Caption
        label = ctk.CTkLabel(self, text="Anmeldung Stahlklassifizierung", font=("Helvetica", 16, "bold"))
        label.grid(row=0, column=0, sticky="ew", pady=40, padx=80)

        # Entry widgets with placeholders
        self.entry_username = ctk.CTkEntry(self, placeholder_text="Username")
        self.entry_username.grid(row=1, column=0, pady=5, padx=100, sticky="ew")
        self.entry_password = ctk.CTkEntry(self, show="*", placeholder_text="Password")
        self.entry_password.grid(row=2, column=0, pady=5, padx=100, sticky="ew")

        self.entry_email = ctk.CTkEntry(self, placeholder_text="Email")
        self.entry_email.grid_forget()
        self.entry_name = ctk.CTkEntry(self, placeholder_text="Full Name")
        self.entry_name.grid_forget()

        # Login Button
        self.btn_login = ctk.CTkButton(self, text="Login", command=self.login)
        self.btn_login.grid(row=5, column=0, pady=5, padx=100, sticky="ew")
        self.bind('<Return>', lambda event: self.login())

        # Go Back Button
        self.btn_back = ctk.CTkButton(self, text="Go back to Login", command=self.login_mode)
        self.btn_back.grid_forget()

        # Registration_mode Button
        self.btn_registration_mode = ctk.CTkButton(self, text="Register", command=self.registration_mode)
        self.btn_registration_mode.grid(row=6, column=0, pady=1, padx=100, sticky="ew")

        # Registration Button
        self.btn_register = ctk.CTkButton(self, text="Register", command=self.register)
        self.btn_register.grid_forget()

        # Reference to the main application
        self.main_app = main_app

    def registration_mode(self):
        self.btn_login.grid_forget()
        self.btn_registration_mode.grid_forget()
        self.btn_back.grid(row=5, column=0, pady=5, padx=100, sticky="ew")
        self.entry_name.grid(row=4, column=0, pady=5, padx=100, sticky="ew")
        self.entry_email.grid(row=3, column=0, pady=5, padx=100, sticky="ew")
        self.btn_register.grid(row=6, column=0, pady=1, padx=100, sticky="ew")
        self.bind('<Return>', lambda event: self.register())

    def login_mode(self):
        self.btn_login.grid(row=5, column=0, pady=5, padx=100, sticky="ew")
        self.btn_registration_mode.grid(row=6, column=0, pady=1, padx=100, sticky="ew")
        self.entry_name.grid_forget()
        self.entry_email.grid_forget()
        self.btn_back.grid_forget()
        self.btn_register.grid_forget()
        self.bind('<Return>', lambda event: self.login())

    def login(self):
        # Get username and password from the entry fields
        username = self.entry_username.get()
        password = self.entry_password.get()

        try:
            login_user(username, password)
            self.destroy()  # Close the login screen
            self.main_app.login_successful = True
        except Exception as e:
            print(e)
            messagebox.showerror("Login Failed", "Invalid username or password")
            self.main_app.login_successful = False

    def register(self):
        # Get values from the entry fields
        username = self.entry_username.get()
        password = self.entry_password.get()
        email = self.entry_email.get()
        name = self.entry_name.get()

        register_user(username, password, email, name)
        messagebox.showinfo("Registration Successful", "You are now registered as: " + username + "!")

        # Close the registration window
        self.login_mode()

    def center_window(self):
        # Get the screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate the window position
        x = (screen_width - self.winfo_reqwidth()) // 2
        y = (screen_height - self.winfo_reqheight()) // 2

        # Set the window position
        self.geometry("+{}+{}".format(x, y))

    def on_close(self):
        stop_fastapi_server()
        self.destroy()