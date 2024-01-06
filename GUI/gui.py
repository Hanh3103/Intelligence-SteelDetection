import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import utils.utils

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


# Main Application class
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # root
        self.title("Steel Identifier.py")
        self.geometry(f"{1100}x{580}")
        self.minsize(500, 300)
        # root Layout
        self.columnconfigure(1, weight=8)
        self.columnconfigure(0, weight=0)
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0, border_width=4)
        self.sidebar_frame.grid(row=0, column=0, rowspan=6, sticky="nsew")

        # Sidebar Widgets
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Side Bar",
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        # Sidebar buttons for different actions
        self.sidebar_button_1 = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='Plot me!',
                                              command=self.update_window)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')

        self.sidebar_button_supervised = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='Supervised',
                                                       command=supervised)
        self.sidebar_button_supervised.grid(row=2, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')

        self.sidebar_button_unsupervised = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='Unsupervised',
                                                         command=unsupervised)
        self.sidebar_button_unsupervised.grid(row=3, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')

        self.sidebar_button_unsupervised = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='registration',
                                                         command=registration)
        self.sidebar_button_unsupervised.grid(row=4, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')

        self.sidebar_button_unsupervised = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='login',
                                                         command=login)
        self.sidebar_button_unsupervised.grid(row=5, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')

        # Sliders for variables in the sidebar
        self.sidebar_slider_frame = ctk.CTkFrame(self.sidebar_frame, width=140)
        self.sidebar_slider_frame.grid(row=6, column=0, padx=20, pady=10, sticky='n')

        self.label_var01 = ctk.CTkLabel(self.sidebar_slider_frame, text="Variable I",
                                        font=ctk.CTkFont(size=10, weight="bold"), anchor='w')
        self.label_var01.grid(row=0, column=0, padx=20, pady=10)
        self.slider_1 = ctk.CTkSlider(self.sidebar_slider_frame, from_=0, to=1, number_of_steps=10)
        self.slider_1.grid(row=0, column=1, padx=20, pady=16, sticky='n')

        self.label_var02 = ctk.CTkLabel(self.sidebar_slider_frame, text="Variable II",
                                        font=ctk.CTkFont(size=10, weight="bold"), anchor='w')
        self.label_var02.grid(row=1, column=0, padx=20, pady=10)
        self.slider_2 = ctk.CTkSlider(self.sidebar_slider_frame, from_=0, to=1, number_of_steps=10)
        self.slider_2.grid(row=1, column=1, padx=20, pady=16, sticky='n')

        self.label_var03 = ctk.CTkLabel(self.sidebar_slider_frame, text="Variable III",
                                        font=ctk.CTkFont(size=10, weight="bold"), anchor='w')
        self.label_var03.grid(row=2, column=0, padx=20, pady=10)
        self.slider_3 = ctk.CTkSlider(self.sidebar_slider_frame, from_=0, to=1, number_of_steps=10)
        self.slider_3.grid(row=2, column=1, padx=20, pady=16, sticky='n')

        # Mainframe creation
        self.main_frame = ctk.CTkFrame(self, )
        self.main_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        # create tabview
        self.tabview = ctk.CTkTabview(self.main_frame, fg_color='transparent')
        self.tabview.grid(row=1, columnspan=2, sticky="nsew")
        self.tabview.add("Plot1")
        self.tabview.tab("Plot1").grid(sticky='nsew')
        self.tabview.add("Plot2")
        self.tabview.add("Plot3")
        self.tabview.tab("Plot1").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Plot1").grid_rowconfigure(1, weight=1)
        self.tabview.tab("Plot2").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Plot3").grid_columnconfigure(0, weight=1)

        # Top right

        # Label for displaying the size of the hovered point
        self.hover_label_var = ctk.StringVar()
        self.hover_label_var.set('Größe des  Punktes:')
        self.hover_label = ctk.CTkLabel(self.main_frame, textvariable=self.hover_label_var)
        self.hover_label.grid(row=0, column=0, padx=10, pady=15, sticky='w')

        # Button to upload a file
        self.import_button = ctk.CTkButton(self.main_frame, corner_radius=8, command=upload_file,
                                           text='Datei Import')
        self.import_button.grid(row=0, column=0, sticky="ne", padx=10, pady=15)

        # Dropdown menu for choosing dark mode
        darkmode_options = ['System', 'Dark', 'Light']
        self.selected_option = tk.StringVar()
        self.selected_option.set(darkmode_options[0])
        self.darkmode_dropdown_menu = ctk.CTkOptionMenu(self.main_frame, values=darkmode_options,
                                                        variable=self.selected_option, command=change_darkmode)
        self.darkmode_dropdown_menu.grid(row=0, column=1, sticky="ne", padx=10, pady=15)

        self.update_window()

    def update_window(self):
        # Function to update the main window content
        plotpoints = 100
        fig, ax = plt.subplots()
        fig.set_size_inches(11, 5.3)
        global x, y, s, c
        x, y, s, c = np.random.rand(4, plotpoints)
        scatter = ax.scatter(x, y, s * 400, c, linewidth=1, edgecolor='black', cmap='viridis')

        # Connect the hover event to update the label
        fig.canvas.mpl_connect('motion_notify_event', lambda event: on_hover(event, scatter, self.hover_label_var))

        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        canvas = FigureCanvasTkAgg(fig, master=self.tabview.tab("Plot1"))
        canvas.draw()
        canvas.get_tk_widget().grid(sticky='nsew', row=1, column=0)
        self.update()


# Function to change the appearance mode (dark mode, light mode, etc.)
def change_darkmode(choice):
    ctk.set_appearance_mode(choice)


# Function to upload a file
def upload_file():
    utils.utils.only_upload_file()


# Function for supervised prediction
def supervised():
    utils.utils.predict_data_from_file_supervised()


# Function for unsupervised prediction
def unsupervised():
    utils.utils.predict_data_from_file_unsupervised()

def registration():
    utils.utils.register_user()

def login():
    utils.utils.login_user()

# Function to handle hover events and update the label
def on_hover(event, scatter, text_label):
    if scatter.contains(event)[0]:
        ind = scatter.contains(event)[1]["ind"][0]

        # Display the size in the text label
        size_of_hovered_point = scatter.get_sizes()[ind]
        text_label.set(f"Größe des  Punktes: {round(size_of_hovered_point, 2)}")
