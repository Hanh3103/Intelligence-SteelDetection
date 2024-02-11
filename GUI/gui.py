import tkinter as tk

import customtkinter as ctk

from api_functions.database import get_all_items_of_user_from_db, upload_csv_or_xlsx_to_db_table, \
    delete_db_table_by_name
from api_functions.supervised import predict_with_supervised_algo
from utils.unsupervised_utils import predict_data_from_file_unsupervised

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

        self.sidebar_button_supervised = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='Supervised')
        self.sidebar_button_supervised.grid(row=2, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')

        self.sidebar_button_unsupervised = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='Unsupervised')
        self.sidebar_button_unsupervised.grid(row=3, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')

        self.sidebar_button_unsupervised = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='get Items',
                                                         command=get_all_items_of_user_from_db_gui)
        self.sidebar_button_unsupervised.grid(row=4, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')


        # Sliders for variables in the sidebar
        self.dynamic_sidebar_frame = ctk.CTkFrame(self.sidebar_frame, width=140)
        self.dynamic_sidebar_frame.grid(row=5, column=0, padx=20, pady=10, sticky='n')

        self.supervised_label = ctk.CTkLabel(self.dynamic_sidebar_frame, text="Supervised",
                                             font=ctk.CTkFont(size=20, weight="bold"))
        self.supervised_label.grid(row=0, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')

        self.check_var = ctk.StringVar(value="False")
        self.checkbox = ctk.CTkCheckBox(master=self.dynamic_sidebar_frame, text="Model is existing",
                                        variable=self.check_var, onvalue="True", offvalue="False", command=self.change_model_status)
        self.checkbox.grid(row=1, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')

        self.dynamic_sidebar_button_Model = ctk.CTkButton(self.dynamic_sidebar_frame, corner_radius=8,
                                                              text='trained Model', state="disabled")
        self.dynamic_sidebar_button_Model.grid(row=3, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')

        self.dynamic_sidebar_button_Dataframe = ctk.CTkButton(self.dynamic_sidebar_frame, corner_radius=8,
                                                              text="Dataframe",
                                                              command=upload_file_to_database)
        self.dynamic_sidebar_button_Dataframe.grid(row=4, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')

        self.dynamic_sidebar_button_encoder = ctk.CTkButton(self.dynamic_sidebar_frame, corner_radius=8,
                                                          text="Labelencoder")
        self.dynamic_sidebar_button_encoder.grid(row=5, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')

        self.algorithm_var = ctk.StringVar(value="xgboost_algorithm")
        self.combobox = ctk.CTkOptionMenu(master=self.dynamic_sidebar_frame,
                                          values=["xgboost_algorithm", "deep_learning"],
                                          command=self.optionmenu_callback)

        self.combobox.grid(row=6, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')

        self.dynamic_sidebar_button_save = ctk.CTkButton(self.dynamic_sidebar_frame, corner_radius=8,
                                                         text='save')
        self.dynamic_sidebar_button_save.grid(row=7, column=0, padx=20, pady=(15, 10))

        self.dynamic_sidebar_button_predict = ctk.CTkButton(self.dynamic_sidebar_frame, corner_radius=8,
                                                         text='predict', command=self.predict_supervised)
        self.dynamic_sidebar_button_predict.grid(row=7, column=1, padx=20, pady=(15, 10))

        self.dynamic_sidebar_button_test = ctk.CTkButton(self.dynamic_sidebar_frame, corner_radius=8,
                                                            text='test')
        self.dynamic_sidebar_button_test.grid(row=7, column=2, padx=20, pady=(15, 10))



        # Mainframe creation
        self.main_frame = ctk.CTkFrame(self, )
        self.main_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.columnconfigure(0, weight=1)


        # Dropdown menu for choosing dark mode
        darkmode_options = ['System', 'Dark', 'Light']
        self.selected_option = tk.StringVar()
        self.selected_option.set(darkmode_options[0])
        self.darkmode_dropdown_menu = ctk.CTkOptionMenu(self.main_frame, values=darkmode_options,
                                                        variable=self.selected_option, command=change_darkmode)
        self.darkmode_dropdown_menu.grid(row=0, column=1, sticky="ne", padx=10, pady=15)

    def optionmenu_callback(self, choice):
        self.algorithm_var.set(choice)

    def change_model_status(self):
        if self.check_var.get() == "True":
            self.dynamic_sidebar_button_Model.configure(state="normal")
            self.dynamic_sidebar_button_Dataframe.configure(state="disabled")
            self.dynamic_sidebar_button_encoder.configure(state="disabled")
        else:
            self.dynamic_sidebar_button_Dataframe.configure(state="normal")
            self.dynamic_sidebar_button_encoder.configure(state="normal")
            self.dynamic_sidebar_button_Model.configure(state="disabled")

    def predict_supervised(self):
        existing_model = self.check_var.get()
        algorithm = self.algorithm_var.get()
        table_name = ""
        if existing_model == "False":
            table_name = "test_labeled_OES_data.csv"
            supervised(existing_model, algorithm, table_name)
        supervised(existing_model, algorithm, table_name)


# Function to change the appearance mode (dark mode, light mode, etc.)
def change_darkmode(choice):
    ctk.set_appearance_mode(choice)


# Function to upload a file
def upload_file_to_database():
    upload_csv_or_xlsx_to_db_table()


# Function for supervised prediction
def supervised(existing_model, algorithm, table_name):
    predict_with_supervised_algo(existing_model,
                                 "",
                                 algorithm,  # please choose between "deep_learning" and
                                 # "xgboost_algorithm"
                                 "test_test.csv",
                                 "prediction")
    # predict_with_supervised_algo("False",
    #                              "hanh3_labeled_OES_data.csv",
    #                              algorithm,
    #                              "hanh3_test.csv",
    #                              "prediction")


# Function for unsupervised prediction
def unsupervised():
    predict_data_from_file_unsupervised()


def get_all_items_of_user_from_db_gui():
    get_all_items_of_user_from_db()


def delete_db_table_by_name_gui():
    delete_db_table_by_name("luca300_labeled_OES_data.csv")



if __name__ == '__main__':
    main_app = App()
    main_app.mainloop()