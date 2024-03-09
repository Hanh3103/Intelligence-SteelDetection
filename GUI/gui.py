import os

from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from GUI.bar_plot import BarPlot
from GUI.supervised_frame import SupervisedFrame
from GUI.unsupervised_frame import UnsupervisedFrame
from api_functions.unsupervised import predict_data_from_file_unsupervised_kmeans
from api_functions.unsupervised import predict_data_from_file_unsupervised_affinity
from api_functions.unsupervised import predict_data_from_file_unsupervised_gmm
from api_functions.unsupervised import predict_data_from_file_unsupervised_hierarchical
from api_functions.unsupervised import predict_data_from_file_unsupervised_tsne
from api_functions.unsupervised import predict_data_from_file_unsupervised_pca
from api_functions.unsupervised import predict_data_from_file_unsupervised_umap

import customtkinter as ctk


import enums.unsupervised_models_enum
from api_functions.database import get_all_items_of_user_from_db, upload_csv_or_xlsx_to_db_table, \
    delete_db_table_by_name
from utils.upload_csv_or_xlsx import return_dataframe_filesize_filename_from_csv_or_xlsx


class App(ctk.CTk):

    # Set appearance mode and default color theme
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    def __init__(self):
        super().__init__()

        # Set login status
        self.login_successful = False

        # root
        self.title("Steel Identifier.py")
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # root Layout
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Mainframe creation
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'resources', 'gui', 'placeholder_diagram.png')
        placeholder_bar_chart_image = ctk.CTkImage(light_image=Image.open(image_path),
                                                   dark_image=Image.open(image_path), size=(500, 500))
        placeholder_label = ctk.CTkLabel(self.main_frame, text='', image=placeholder_bar_chart_image)
        placeholder_label.grid(row=0, column=0, sticky="nsew")
        self.canvas = None

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, corner_radius=0, width=300)
        self.sidebar_frame.grid(row=0, column=0, sticky="ns")
        self.sidebar_frame.grid_propagate(False)
        self.sidebar_frame.columnconfigure(0, weight=1)

        '''Creation: Sidebar elements'''
        self.static_sidebar_frame = ctk.CTkFrame(self.sidebar_frame)
        self.dynamic_sidebar_frame = ctk.CTkFrame(self.sidebar_frame)

        '''Grid Config: Sidebar elements'''
        self.static_sidebar_frame.columnconfigure(0, weight=1)
        self.static_sidebar_frame.grid(row=0, column=0, padx=(0, 6), pady=(0, 12), sticky='nsew')
        self.dynamic_sidebar_frame.grid(row=1, column=0, padx=(0, 6), sticky='nsew')
        self.dynamic_sidebar_frame.columnconfigure(0, weight=1)

        #Static  Sidebar
        '''Creation: static elements'''
        self.logo_label = ctk.CTkLabel(self.static_sidebar_frame, text="Side Bar",
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.sidebar_button_supervised = ctk.CTkButton(self.static_sidebar_frame, corner_radius=8, text='Supervised',
                                                       command=self.show_supervised)
        self.sidebar_button_unsupervised = ctk.CTkButton(self.static_sidebar_frame, corner_radius=8,
                                                         text='Unsupervised', command=self.show_unsupervised)
        self.sidebar_button_upload = ctk.CTkButton(self.static_sidebar_frame, corner_radius=8,
                                                   text='upload Dataframe to Database',
                                                   command=upload_file_to_database)

        '''Grid Config: static elements'''
        self.logo_label.grid(row=0, column=0, padx=6, pady=(12, 6))
        self.sidebar_button_supervised.grid(row=1, column=0, padx=20, pady=6, sticky='nsew')
        self.sidebar_button_unsupervised.grid(row=2, column=0, padx=20, pady=6, sticky='nsew')
        self.sidebar_button_upload.grid(row=3, column=0, padx=20, pady=6, sticky='nsew')

        # Dynamic Sidebar
        # ------------------ Unsupervised ------------------------------

        self.unsupervised_frame = UnsupervisedFrame(self.dynamic_sidebar_frame, self)

        # ------------------ Supervised ---------------------------------

        self.supervised_frame = SupervisedFrame(self.dynamic_sidebar_frame, self)
        self.supervised_frame.grid(row=1, column=0, sticky='nsew')


    '''calls the method: return_dataframe_filesize_filename_from_csv_or_xlsx() and saves return file in var: 
    unsupervised_data'''
    def upload_file(self):
        self.unsupervised_data = (return_dataframe_filesize_filename_from_csv_or_xlsx())

    '''This method creates a canvas to display a bar plot of element percentages in various steel types using the 
    provided predicted steel types and prediction data'''
    def create_canvas(self, predicted_stell_types, prediction_data):
        bar_plot = BarPlot()
        elemente = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Ni', 'Al', 'Cu']
        fig = bar_plot.create_supervised_Plot(predicted_stell_types, elemente, prediction_data)
        self.canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(relx=0.0, rely=0.0, width=self.main_frame.winfo_width(),
                                          height=self.main_frame.winfo_height())

    '''updates the unsupervised_slider_label with the current value of the slider'''
    def sliding(self):
        self.unsupervised_slider_label.configure(text=str(self.unsupervised_slider_var.get()))

    '''displays the supervised frame in the dynamic part of the Sidebar and hides the unsupervised part'''
    def show_supervised(self):
        if self.unsupervised_frame.winfo_ismapped():
            self.supervised_frame.grid(row=0, column=0, sticky='nsew')
            self.unsupervised_frame.grid_forget()

    '''displays the unsupervised frame in the dynamic part of the Sidebar and hides the supervised part'''
    def show_unsupervised(self):
        if self.supervised_frame.winfo_ismapped():
            self.unsupervised_frame.grid(row=0, column=0, sticky='nsew')
            self.supervised_frame.grid_forget()

    def option_menu_callback_algorithm(self, choice):
        self.unsupervised_algorithm_var.set(choice)

    def predict_unsupervised(self):

        """
        When the Button 'Predict' in the category unsupervised is pressed it collects the algorithm which the user
        choose, the datafile and the different arguments which differ with each algorithm, to call the respected
        algorithm.
        """
        algorithm = enums.unsupervised_models_enum.Models.get_enum_by_value(self.unsupervised_algorithm_combobox.get())
        file = self.unsupervised_data
        komponents = self.unsupervised_slider_var.get()
        unsupervised(algorithm, file, komponents)

    def on_close(self):
        self.destroy()


    # Function to change the appearance mode (dark mode, light mode, etc.)
def change_dark_mode(choice):
    ctk.set_appearance_mode(choice)


# Function to upload a file
def upload_file_to_database():
    upload_csv_or_xlsx_to_db_table()


# Function for unsupervised prediction
def unsupervised(model, file, komponents):
    if model == enums.unsupervised_models_enum.Models.KMEANS:
        predict_data_from_file_unsupervised_kmeans(file, komponents)
    elif model == enums.unsupervised_models_enum.Models.GMM:
        predict_data_from_file_unsupervised_gmm(file, komponents)
    elif model == enums.unsupervised_models_enum.Models.hierarchical:
        predict_data_from_file_unsupervised_hierarchical(file, komponents)
    elif model == enums.unsupervised_models_enum.Models.tsne:
        predict_data_from_file_unsupervised_tsne(file, ...)
    elif model == enums.unsupervised_models_enum.Models.pca:
        predict_data_from_file_unsupervised_pca(file)
    elif model == enums.unsupervised_models_enum.Models.umap:
        predict_data_from_file_unsupervised_umap(file, ...)
    elif model == enums.unsupervised_models_enum.Models.affinity:
        predict_data_from_file_unsupervised_affinity(file)
        
        
def get_all_items_of_user_from_db_gui():
    return get_all_items_of_user_from_db()


def delete_db_table_by_name_gui():
    delete_db_table_by_name("luca300_labeled_OES_data.csv")


