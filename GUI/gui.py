import tkinter as tk

import customtkinter as ctk
import utils.utils


# Set appearance mode and default color theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Main Application class
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Set login status
        self.login_successful = False

        # root
        self.title("Steel Identifier.py")
        self.geometry(f"{1500}x{800}")


        # Bind the Escape key to exit fullscreen
        self.bind('<Escape>', lambda event: self.attributes('-fullscreen', False))

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
        self.sidebar_button_1 = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='Plot me!')
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')

        self.sidebar_button_supervised = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='Supervised',
                                                       command=supervised)
        self.sidebar_button_supervised.grid(row=2, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')

        self.sidebar_button_unsupervised = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='Unsupervised')
        self.sidebar_button_unsupervised.grid(row=3, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')

        self.sidebar_button_unsupervised = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='get Items',
                                                         command=get_user_items)
        self.sidebar_button_unsupervised.grid(row=4, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')


        # Sliders for variables in the sidebar
        self.dynamic_sidebar_frame = ctk.CTkFrame(self.sidebar_frame, width=140)
        self.dynamic_sidebar_frame.grid(row=5, column=0, padx=20, pady=10, sticky='n')

        self.supervised_label = ctk.CTkLabel(self.dynamic_sidebar_frame, text="Supervised",
                                             font=ctk.CTkFont(size=20, weight="bold"))
        self.supervised_label.grid(row=0, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')

        self.check_var = ctk.StringVar(value="off")
        self.checkbox = ctk.CTkCheckBox(master=self.dynamic_sidebar_frame, text="Model is existing",
                                        variable=self.check_var, onvalue="on", offvalue="off", command=self.change_model_status)
        self.checkbox.grid(row=1, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')

        self.dynamic_sidebar_button_Model = ctk.CTkButton(self.dynamic_sidebar_frame, corner_radius=8,
                                                              text='trained Model', state="disabled")
        self.dynamic_sidebar_button_Model.grid(row=3, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')

        self.dynamic_sidebar_button_Dataframe = ctk.CTkButton(self.dynamic_sidebar_frame, corner_radius=8,
                                                              text="Dataframe",
                                                              command=unsupervised)
        self.dynamic_sidebar_button_Dataframe.grid(row=4, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')

        self.dynamic_sidebar_button_encoder = ctk.CTkButton(self.dynamic_sidebar_frame, corner_radius=8,
                                                          text="Labelencoder")
        self.dynamic_sidebar_button_encoder.grid(row=5, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')


        self.combobox = ctk.CTkOptionMenu(master=self.dynamic_sidebar_frame,
                                          values=["XGBoost", "Deep Learning"],
                                          command=self.optionmenu_callback)
        self.combobox.grid(row=6, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')

        self.dynamic_sidebar_button_save = ctk.CTkButton(self.dynamic_sidebar_frame, corner_radius=8,
                                                         text='save')
        self.dynamic_sidebar_button_save.grid(row=7, column=0, padx=20, pady=(15, 10))

        self.dynamic_sidebar_button_predict = ctk.CTkButton(self.dynamic_sidebar_frame, corner_radius=8,
                                                         text='predict')
        self.dynamic_sidebar_button_predict.grid(row=7, column=1, padx=20, pady=(15, 10))

        self.dynamic_sidebar_button_test = ctk.CTkButton(self.dynamic_sidebar_frame, corner_radius=8,
                                                            text='test')
        self.dynamic_sidebar_button_test.grid(row=7, column=2, padx=20, pady=(15, 10))



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

        # Dropdown menu for choosing dark mode
        darkmode_options = ['System', 'Dark', 'Light']
        self.selected_option = tk.StringVar()
        self.selected_option.set(darkmode_options[0])
        self.darkmode_dropdown_menu = ctk.CTkOptionMenu(self.main_frame, values=darkmode_options,
                                                        variable=self.selected_option, command=change_darkmode)
        self.darkmode_dropdown_menu.grid(row=0, column=1, sticky="ne", padx=10, pady=15)


    def optionmenu_callback(choice):
        print("optionmenu dropdown clicked:", choice)

    def change_model_status(self):
        if self.check_var.get() == "on":
            self.dynamic_sidebar_button_Model.configure(state="normal")
            self.dynamic_sidebar_button_Dataframe.configure(state="disabled")
            self.dynamic_sidebar_button_encoder.configure(state="disabled")
        else:
            self.dynamic_sidebar_button_Dataframe.configure(state="normal")
            self.dynamic_sidebar_button_encoder.configure(state="normal")
            self.dynamic_sidebar_button_Model.configure(state="disabled")



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



def get_user_items():
    utils.utils.get_all_items()


if __name__ == '__main__':
    main_app = App()
    main_app.mainloop()