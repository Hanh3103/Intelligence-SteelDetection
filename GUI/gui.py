import os

from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from GUI.bar_plot import BarPlot
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
from api_functions.supervised import predict_with_supervised_algo
from utils.database_utils import get_dataframe_from_database
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
        '''Creation: unsupervised elements'''
        self.unsupervised_frame = ctk.CTkFrame(self.dynamic_sidebar_frame, fg_color='#333333')
        self.unsupervised_frame.columnconfigure(0, weight=1)

        self.Unsupervised_label = ctk.CTkLabel(self.unsupervised_frame, text="Unsupervised",
                                               font=ctk.CTkFont(size=20, weight="bold"))

        unsupervised_algorithms = enums.unsupervised_models_enum.Models.create_list(self)
        self.unsupervised_algorithm_var = ctk.StringVar(value=unsupervised_algorithms[0])
        self.unsupervised_algorithm_combobox = ctk.CTkOptionMenu(master=self.unsupervised_frame,
                                                                 values=unsupervised_algorithms,
                                                                 command=self.option_menu_callback_algorithm)
        self.unsupervised_algorithm_combobox.set(unsupervised_algorithms[0])

        self.unsupervised_slider_var = ctk.IntVar(value=5)
        self.unsupervised_slider = ctk.CTkSlider(master=self.unsupervised_frame, from_=0, to=15,
                                                 variable=self.unsupervised_slider_var, command=self.sliding)
        self.unsupervised_slider_label = ctk.CTkLabel(master=self.unsupervised_frame,
                                                      text=str(self.unsupervised_slider_var.get()))

        self.unsupervised_button_upload = ctk.CTkButton(self.unsupervised_frame, corner_radius=8,
                                                         text='upload File', command=self.upload_file)
        self.unsupervised_button_predict = ctk.CTkButton(self.unsupervised_frame, corner_radius=8,
                                                       text='predict', command=self.predict_unsupervised)


        self.unsupervised_data = ''

        '''Grid Config: unsupervised elements'''
        self.Unsupervised_label.grid(row=0, column=0, padx=20, pady=(15, 10), columnspan=3, sticky='ew')
        self.unsupervised_algorithm_combobox.grid(row=1, column=1, padx=20, pady=6, columnspan=3, sticky='ew')
        self.unsupervised_slider.grid(row=2, column=1, padx=20, pady=(6, 0), sticky='ew')
        self.unsupervised_slider_label.grid(row=3, column=1, padx=20, sticky='ew')
        self.unsupervised_button_upload.grid(row=7, column=1, padx=20, pady=6, sticky='ew')
        self.unsupervised_button_predict.grid(row=8, column=1, padx=20, pady=6, sticky='ew')

        # ------------------ Supervised ---------------------------------
        '''Creation: supervised elements'''
        self.supervised_frame = ctk.CTkFrame(self.dynamic_sidebar_frame, fg_color='#333333')
        self.supervised_frame.grid(row=0, column=0, sticky='nsew')
        self.supervised_frame.columnconfigure(0, weight=1)

        self.supervised_label = ctk.CTkLabel(self.supervised_frame, text="Supervised",
                                             font=ctk.CTkFont(size=20, weight="bold"))

        self.supervised_model_var = ctk.StringVar(value="False")
        self.supervised_model_checkbox = ctk.CTkCheckBox(master=self.supervised_frame, text="Model is existing",
                                                         variable=self.supervised_model_var, onvalue="True",
                                                         offvalue="False", command=self.change_model_status)

        self.supervised_button_Model = ctk.CTkButton(self.supervised_frame, corner_radius=8,
                                                     text='trained Model', state="disabled")

        self.supervised_button_Dataframe = ctk.CTkButton(self.supervised_frame, corner_radius=8,
                                                         text="Dataframe", state="disabled")

        self.supervised_button_encoder = ctk.CTkButton(self.supervised_frame, corner_radius=8,
                                                       text="Labelencoder", state="disabled")

        self.unsupervised_algorithm_var = ctk.StringVar(value="xgboost_algorithm")
        self.supervised_algorithm_combobox = ctk.CTkOptionMenu(master=self.supervised_frame,
                                                               values=["xgboost_algorithm", "deep_learning"],
                                                               command=self.option_menu_callback_algorithm)

        dataframes = list(get_all_items_of_user_from_db_gui())
        self.dataframe_var = ctk.StringVar(value=dataframes[0])
        self.dataframe_combobox = ctk.CTkOptionMenu(master=self.supervised_frame,
                                                    values=dataframes,
                                                    command=self.option_menu_callback_dataframe)

        self.supervised_button_predict = ctk.CTkButton(self.supervised_frame, corner_radius=8,
                                                       text='predict', command=self.predict_supervised)

        '''Grid Config: supervised elements'''
        self.supervised_label.grid(row=0, column=0, padx=20, pady=(15, 10), sticky='ew')
        self.supervised_model_checkbox.grid(row=1, column=0, padx=20, pady=6, sticky='ew')
        self.supervised_button_Model.grid(row=7, column=0, padx=20, pady=6, sticky='ew')
        self.supervised_button_Dataframe.grid(row=4, column=0, padx=20, pady=6, sticky='ew')
        self.supervised_button_encoder.grid(row=5, column=0, padx=20, pady=6, sticky='ew')
        self.supervised_algorithm_combobox.grid(row=2, column=0, padx=20, pady=6, sticky='ew')
        self.dataframe_combobox.grid(row=3, column=0, padx=20, pady=6, sticky='ew')
        self.supervised_button_predict.grid(row=8, column=0, padx=20, pady=6, sticky='ew')

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

    def option_menu_callback_dataframe(self, choice):
        self.dataframe_var.set(choice)

    def change_model_status(self):
        if self.supervised_model_var.get() == "True":
            self.supervised_button_Model.configure(state="normal")
            self.supervised_button_Dataframe.configure(state="disabled")
            self.supervised_button_encoder.configure(state="disabled")
        else:
            self.supervised_button_Dataframe.configure(state="normal")
            self.supervised_button_encoder.configure(state="normal")
            self.supervised_button_Model.configure(state="disabled")

    def predict_supervised(self):
        existing_model = self.supervised_model_var.get()
        algorithm = self.unsupervised_algorithm_var.get()
        dataframe_name = self.dataframe_var.get()
        if existing_model == "True":
            prediction = predict_with_supervised_algo(
                existing_model,
                "",
                algorithm,
                dataframe_name,
                "prediction")
        else:
            prediction = predict_with_supervised_algo(
                existing_model,
                "test_labeled_OES_data.csv",
                algorithm,
                dataframe_name,
                "prediction")
        prediction_data = get_dataframe_from_database(dataframe_name)
        self.create_canvas(prediction, prediction_data)
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


